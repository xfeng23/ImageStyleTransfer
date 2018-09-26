# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.shortcuts import render
from django.template import loader
from django.http import HttpResponse, JsonResponse, HttpResponseBadRequest, HttpResponseNotFound, HttpResponseServerError
from rest_framework.decorators import api_view
from io import BytesIO
from PIL import Image
from fast_neural_style.django.StyleTransferEndpoint import backendGlobal
from fast_neural_style.helpers import *
from fast_neural_style.config import *
import multiprocessing
import os, sys, time, uuid
import requests, random, re
import threading
import torch


# Html view
def tester(request):
    return render(request, 'StyleTransferEndpoint/styletransfer_tester.html') 

# Create your views here.

@api_view(['GET'])
def index(request):
    return HttpResponse("Images are stylized by these apis.")

@api_view(['GET'])
def tester_stylize(request):
    url = request.GET.get('url')
    style = request.GET.get('style')
    # get style from disk
    production_styles = get_style_from_disk()
    if style not in production_styles:
        style = DEFAULT_STYLE
    style_path = os.path.join(MODELS_DIR, style) + '.pth'
    img = load_image_from_url(url)
    if img is None:
        return HttpResponseBadRequest("Bad Url")
    img = preprocess_image(img, MAX_DIM)
    img = utils.stylize(img, style_path, USE_GPU, VERBOSE=True)
    if img is None:
        return HttpResponseServerError("internal server error")
    img_str = image_to_memory_stream(img)
    return HttpResponse(img_str, content_type="text/plain")


# Will be deprecated by below Django endpoints
@api_view(['GET'])
def stylize(request):
    style_path = os.path.join(MODELS_DIR, random.choice(os.listdir(MODELS_DIR)))
    url = request.GET.get('url')

    img = load_image_from_url(url)
    if img is None:
        return HttpResponseBadRequest("Bad Url")
    
    img = preprocess_image(img, MAX_DIM)
    img = utils.stylize(img, style_path, USE_GPU, VERBOSE=True)
    
    blob_url = upload_image_to_blob(img)

    style_name = re.match('.*/([\s|\w]+)\.pth', style_path).group(1)
    sticker_name = "Picasso (" + style_name + ")"
    print(sticker_name)
    # return as json object
    return JsonResponse({'Imgurl': blob_url, 'StickerName': sticker_name})


"""
GET request.
Takes in an image url and returns memoryStream of stylized 
result.

The style is randomly picked by the server.
=========================
Parameters:
    - request: incomming request

Return:
    - memory stream of stylized image
"""
@api_view(['GET'])
def generate_stylized_memory_stream(request):
    try:
        
        view_start = time.time()
        host_ip = request.get_host().split(':')[0]
        decodedUrl = request.GET.get('url')
        style_path = os.path.join(MODELS_DIR, random.choice(os.listdir(MODELS_DIR)))
        request.appinsights.client.track_trace('{}|{} style selected: {}'.format(decodedUrl, host_ip, style_path)) #log     
        img = load_image_from_url(decodedUrl, request.appinsights.client)
        
        if img is None:
            request.appinsights.client.track_trace('{}|{} load image from url returns None.'.format(decodedUrl, host_ip)) #log
            return HttpResponseBadRequest("Bad image url.")
        else:
            request.appinsights.client.track_trace('{}|{} original image size: {}'.format(decodedUrl, host_ip, img.size)) #log
            
        img = utils.stylize(img, request.appinsights.client, style_path, backendGlobal.style_model, 
                            backendGlobal.gpuHandle, host_ip, decodedUrl, USE_GPU)

        if img is None:
            request.appinsights.client.track_trace('{}|{} stylize returns None.'.format(decodedUrl, host_ip)) #log
            return HttpResponseServerError("Stylization fails.")
        else:
            request.appinsights.client.track_trace('{}|{} stylized image size: {}'.format(decodedUrl, host_ip, img.size)) #log

        img_stream = image_to_memory_stream(img, request.appinsights.client)
        if img_stream is None:
            request.appinsights.client.track_trace('{}|{} image to memory stream returns None.')
            return HttpResponseServerError("Convert image to memory stream fails.")
        else:
            request.appinsights.client.track_trace('{}|{} image stream length: {}'.format(decodedUrl, host_ip, len(img_stream))) #log

        style_name = re.match('.*/([\s|\w]+)\.pth', style_path).group(1)
        sticker_name = "Picasso (" + style_name + ")"
        # return as json object
        return JsonResponse({
			     'Data': img_stream, 
			     'StickerName': sticker_name, 
			    })
    except Exception as e:
        request.appinsights.client.track_trace('{}|{} exception happen in backend services: {}'.format(decodedUrl, host_ip, e)) #log
        request.appinsights.client.track_exception()
        return HttpResponseServerError("backend services error")
    finally:
        backend_service_runtime = (time.time() - view_start) * 1000
        request.appinsights.client.track_trace('{}|{} backend services runtime: {}'.format(decodedUrl, host_ip, backend_service_runtime)) #log


"""
GET request.
Takes in an image url and returns corresponding uuid. 
Begins a new thread to stylize image and upload stylized
image to azure blob storage with identifier uuid.jpg

The style is randomly picked by the server. 
=========================
Parameters:
    - url: url of image

Return:
    - uuid: corresponding to the eventual stylized image
"""
@api_view(['GET'])
def generate_image_uuid_url(request):
    url = request.GET.get('url')

    img = load_image_from_url(url)
    if img is None:
        return HttpResponseBadRequest("Bad Url")

    blob_id = str(uuid.uuid4())

    args = (img, blob_id)
    thread.start_new_thread(stylize_and_upload, args)

    return JsonResponse({"uuid": blob_id})


"""
GET request

Parameters:
    - blob_url 'str': must be in format 'UUID.jpg'

Return:
    - JPG HttpResponse: returns byte array of stylized image

Return:
    - JPG HttpResponse: returns byte array of stylized image
Parameters:
    - blob_url 'str': must be in format 'UUID.jpg'

Return:
    - JPG HttpResponse: returns byte array of stylized image
=========================

Takes in 'UUID.jpg' corresponding to stylized image in azure blob storage.
Image may not be in blob storage when GET request is initiated.
Loops for 10 seconds checking for the image. 
If not found, returns HTTP error. 
"""
# GET http://localhost:8000/api/images/01367274-b27c-4e34-b547-9dd93dc4fef8.jpg
@api_view(['GET'])
def get_image_from_uuid(request, blob_url):
    img_url = make_blob_url(blob_url)
    print("img_url", img_url)
    t = 0.0
    retry_interval = 0.1
    while True:
        r = requests.get(img_url)
        if r.status_code == 200:
            break
        time.sleep(retry_interval)
        t += retry_interval
        # if 10 seconds pass, return not found
        if t > 10:
            return HttpResponseNotFound()

    try:
        bytes = BytesIO(r.content)
    except:
        print("Error caught in load_image_from_url")
        return HttpResponseBadRequest("Url does not correspond to image")
    return HttpResponse(bytes, content_type="image/jpeg")

@api_view(['GET'])
def get_styles(request):
    styles = get_style_from_disk()
    return JsonResponse(styles, safe=False)
