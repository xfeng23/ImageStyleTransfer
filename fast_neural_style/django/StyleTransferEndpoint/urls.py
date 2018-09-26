"""mysite URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import include, url

from StyleTransferEndpoint import views

"""
urlpatterns = [
    url(r'^api/', include('api.urls')),
]
"""

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^api/images/(?P<blob_url>[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\.jpg)/', views.get_image_from_uuid, name='get_image_from_uuid'),
    url(r'^api/stylize_url/', views.generate_image_uuid_url, name='generate_image_uuid_url'),
    #url(r'^api/stylize_data/', views.generate_image_uuid_data, name='generate_image_uuid_data'),
    url(r'^api/blob/stylize/', views.stylize, name='stylize'),
    url(r'^api/stylize/', views.generate_stylized_memory_stream, name='generate_memory_stream_from_url'),
    url(r'^api/tester_stylize/', views.tester_stylize, name='tester_stylize'),
    url(r'^api/styles/', views.get_styles, name='get_styles'),
    url(r'^api/styletransfer_tester/', views.tester, name='tester')
    #url(r'^api/styletransfer_trainer/upload/', views.upload_style, name='upload_styles')
]
