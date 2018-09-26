from django.shortcuts import render
from django.template import loader
from django.http import HttpResponseRedirect, HttpResponse, HttpResponseServerError
from rest_framework.decorators import api_view
import urllib
import requests, json
import multiprocessing
import threading
import time
# Aria
from fast_neural_style.aria_utilities.aria import EventProperties
from fast_neural_style.aria_utilities.aria_helper import *

SUCCESS_CODE = 200
INTERNAL_ERROR_CODE = 500

# the capacity of semaphore
SEMA = threading.Semaphore(10)

# counter used to distribute requests
REQUEST_ROUTER = 0

# multiprocessing lock
l = multiprocessing.Lock()

# URL_SERVICE
BACKEND_SERVICE_PORT = ['30006', '33100', '31004', '35010']

@api_view(['GET'])
def distribute(request):
    """ distribute requests to different backend services """
    global REQUEST_ROUTER, SEMA
    event_properties = None
    decodedUrl = request.GET.get('url')
    host_ip = request.get_host().split(':')[0]
    # check semaphore
    if not SEMA.acquire(False):
        request.appinsights.client.track_trace('{}|{} semaphore cannot acquired.'.format(decodedUrl, host_ip)) #log
        request.appinsights.client.track_trace('{}|{} request distributor over loaded.'.format(decodedUrl, host_ip)) #log
        return HttpResponseServerError("internal server error.")
    else:
        request.appinsights.client.track_trace('{}|{} semaphore acquired.'.format(decodedUrl, host_ip)) #log
    # initialize aria
    event_properties = EventProperties('stratus_orders')
    logger, log_manager = init_aria_log()
    try:

        distribute_start = time.time()
        local_router = 0
        try:
            l.acquire()
            request.appinsights.client.track_trace('{}|{} distributor lock acquired.'.format(decodedUrl, host_ip)) #log
            REQUEST_ROUTER = 0 if REQUEST_ROUTER == 3 else REQUEST_ROUTER + 1
            local_router = REQUEST_ROUTER
            request.appinsights.client.track_trace('{}|{} local router: {}'.format(decodedUrl, host_ip, local_router)) #log
        finally:
            l.release()
            request.appinsights.client.track_trace('{}|{} distributor lock released.'.format(decodedUrl, host_ip)) #log

        # encoding url
        encodedUrl = urllib.parse.quote_plus(decodedUrl)
        backendResponse = requests.get('http://{}:{}/api/stylize/?url={}'.format(host_ip, BACKEND_SERVICE_PORT[local_router], encodedUrl))
        if backendResponse.status_code != SUCCESS_CODE:
            request.appinsights.client.track_trace('{}|{} backend service status code: {} message: {}'.format(decodedUrl, host_ip, backendResponse.status_code, backendResponse.text)) #log
            return HttpResponseServerError("internal server error.")
        else:
            request.appinsights.client.track_trace('{}|{} backend service status code: 200'.format(decodedUrl, host_ip))
        request.appinsights.client.track_trace('{}|{} backend service result length: {}'.format(decodedUrl, host_ip, len(backendResponse.content))) #log
        
        # aria
        event_properties.set_property('{}|{} response code'.format(decodedUrl, host_ip), '{}'.format(backendResponse.status_code))
        event_properties.set_property('{}|{} response length'.format(decodedUrl, host_ip), '{}'.format(len(backendResponse.content)))
        send_aria_log(logger, log_manager, event_properties)

        return HttpResponse(backendResponse.content, content_type="application/json")
    except Exception as e:
        request.appinsights.client.track_trace('{}|{} exception happen in distributor: {}'.format(decodedUrl, host_ip, e)) #log
        request.appinsights.client.track_exception()
        event_properties.set_property('{}|{} response code'.format(decodedUrl, host_ip), '{}'.format(INTERNAL_ERROR_CODE)) # log error code
        event_properties.set_property('{}|{} exception'.format(decodedUrl, host_ip), '{}'.format(e)) # log exception
        send_aria_log(logger, log_manager, event_properties)
        return HttpResponseServerError("internal server error.")
    finally:
        
        distribute_time = (time.time() - distribute_start) * 1000
        request.appinsights.client.track_trace('{}|{} distribute time: {}'.format(decodedUrl, host_ip, distribute_time)) #log
        SEMA.release()
        request.appinsights.client.track_trace('{}|{} semaphore released.'.format(decodedUrl, host_ip)) #log