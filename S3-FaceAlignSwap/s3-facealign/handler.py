try: 
    import unzip_requirements
    #from imagenet_labels import  imagenet_labels
except ImportError: 
    print("Importing unzip_requirements  failed")
    pass 
import boto3
import os
import io 
import json 
import base64 
import cv2
import dlib
import faceBlendCommon as fbc
from  PIL import Image
import numpy as np
from renderFace import renderFace, renderFace2
from requests_toolbelt.multipart import decoder
from requests_toolbelt import MultipartEncoder


print( "Import End....")

S3_BUCKET = os.environ['S3_BUCKET'] if 'S3_BUCKET' in os.environ else 'rekog-eva4s1'
PREDICTOR_PATH = os.environ['PREDICTOR_PATH'] if 'PREDICTOR_PATH' in os.environ else 'shape_predictor_68_face_landmarks.dat'

s3 = boto3.client('s3')
try: 
    if os.path.isfile( PREDICTOR_PATH) != True:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=PREDICTOR_PATH)
        print( "Creating Bytestream") 
        bytestream = io.BytesIO(obj['Body'].read() ) 
        print("Loading Detectors")
        temporarylocation="/tmp/shape_predictor_68_face_landmarks.dat"
        with open(temporarylocation,'wb') as out: ## Open temporary file as bytes
                out.write(bytestream.read())   
        faceDetector = dlib.get_frontal_face_detector()
        #landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)
        landmarkDetector = dlib.shape_predictor(temporarylocation)
        print( "Detectors Loaded")
except Exception as e:
    print("Dlib loading failed!",repr(e))
    raise(e)


def perform_face_alignment(image_bytes):
    h=w=600
    points = fbc.getLandmarks(faceDetector, landmarkDetector, image_bytes)
    points = np.array(points)
    image_bytes = np.float32(image_bytes)/255.0
    imNorm, points = fbc.normalizeImagesAndLandmarks((h,w), image_bytes, points)
    imNorm = np.uint8(imNorm * 255)
    return imNorm[:,:,::-1]


def detect_face_presence( image_bytes ) : 
    #im = np.array(Image.open(io.BytesIO(image_bytes)))
    faceRects = faceDetector(image_bytes, 0)
    print("Number of faces detected:{}".format(len(faceRects)))
    return faceRects

def classify_image(event, context) : 
    try: 
        content_type_header = event['headers']['content-type'] 
        #print(event['body'])
        body = base64.b64decode(event["body"]) 
        print( 'BODY LOADED' )
        picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
        image_bytes = np.array(Image.open(io.BytesIO(picture.content)))
        faceCount =  detect_face_presence(image_bytes = image_bytes) 

        content_disp = picture.headers[b'Content-Disposition'].decode().split(';')
        filename = content_disp[1].split('=')[1]
        if len(filename) < 4:
            filename = content_disp[2].split('=')[1]

        if(len(faceCount) < 1):
            print("No faces detected hence no alignment will be done")
            return { 
                "statusCode": 200, 
                "headers":{ 
                    'Content-Type': 'application/json' , 
                    'Access-control-Allow-origin': '*', 
                    'Access-control-Allow-credentials' : True
                },
                "body":json.dumps({'file':filename.replace('"',''), 'facecount': len(faceCount) })
            }

        aligned_face = perform_face_alignment(image_bytes= image_bytes) 
        #predicted_str = f'That seems to be {predicted_val}'


        ### We will return a base64 encoded JPG file so will set the content type etc accordingly
        resultant_image = base64.b64encode(cv2.imencode('.jpg', aligned_face)[1])
        #multipart_encoded_reply =  MultipartEncoder({'field2': ('filename', resultant_image, 'image/jpg')})
        
        return { 
            "statusCode": 200, 
            "headers":{ 
                'Content-Type': 'application/json',#'image/jpg', #multipart_encoded_reply.content_type , 
                'Access-control-Allow-origin': '*', 
                'Access-control-Allow-credentials' : True
            },
            "body": json.dumps({'file':filename.replace('"',''), 'facecount': len(faceCount),'image':resultant_image.decode()})#multipart_encoded_reply.to_string()
        }
    except Exception as e:
        print("classify_image Exception", repr(e))
        return { 
            "statusCode": 500,
            "headers":{
                'Content-Type': 'application/json',
                'Access-control-Allow-origin':'*',
                "Access-control-Allow-credentials":True
            },
            "body": json.dumps({"error": repr(e)})
        }
