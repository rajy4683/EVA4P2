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
import re


print( "Import End....")

S3_BUCKET = os.environ['S3_BUCKET'] if 'S3_BUCKET' in os.environ else 'rekog-eva4s1'
PREDICTOR_PATH = os.environ['PREDICTOR_PATH'] if 'PREDICTOR_PATH' in os.environ else 'shape_predictor_68_face_landmarks.dat'
IMAGE_DIR = os.environ['IMAGE_DIR'] if 'IMAGE_DIR' in os.environ else 'baseimages/'

s3 = boto3.client('s3')
s3_res = boto3.resource('s3')
bucket = s3_res.Bucket(S3_BUCKET)

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
        ###### Load main folder of images
        obj_list = [(object_summary.key, os.path.basename(object_summary.key).split(".")[0][:-1]) for object_summary in bucket.objects.filter(Prefix="baseimages/") if re.match(".*.jpg", object_summary.key) ]
        print("Objects found:{}".format(obj_list))

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
    return imNorm#[:,:,::-1]


def detect_face_presence( image_bytes ) : 
    #im = np.array(Image.open(io.BytesIO(image_bytes)))
    faceRects = faceDetector(image_bytes, 0)
    print("Number of faces detected:{}".format(len(faceRects)))
    return faceRects

def select_random_body():
    rand_val = np.random.randint(0,len(obj_list)-1)
    my_selected_obj = bucket.Object(obj_list[rand_val][0])
    myimg_read = io.BytesIO(my_selected_obj.get()['Body'].read())
    return Image.open(myimg_read), , obj_list[rand_val][1]

def execute_face_swap(img1, img2):
    im1Display = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    im2Display = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    img1Warped = np.copy(img2)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    # Read array of corresponding points
    points1 = fbc.getLandmarks(detector, predictor, img1)
    points2 = fbc.getLandmarks(detector, predictor, img2)
    hullIndex = cv2.convexHull(np.array(points2), returnPoints=False)
    if(len(points1) == 0 or len(points2) == 0):
        print("Landmark detection failed for selected Images Source:{} Dest:{}".format(len(points1), len(points)))

    # Create convex hull lists
    hull1 = []
    hull2 = []
    for i in range(0, len(hullIndex)):
        hull1.append(points1[hullIndex[i][0]])
        hull2.append(points2[hullIndex[i][0]])
    hull8U = []
    for i in range(0, len(hull2)):
        hull8U.append((hull2[i][0], hull2[i][1]))

        mask = np.zeros(img2.shape, dtype=img2.dtype)
        cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))

            # Find Centroid
        m = cv2.moments(mask[:,:,1])
        center = (int(m['m10']/m['m00']), int(m['m01']/m['m00']))
    
    sizeImg2 = img2.shape
    rect = (0, 0, sizeImg2[1], sizeImg2[0])
    dt = fbc.calculateDelaunayTriangles(rect, hull2)

    # If no Delaunay Triangles were found, quit
    if len(dt) == 0:
        #quit()
        print("No Delaunay Triangles were found!")
        return None
    imTemp1 = im1Display.copy()
    imTemp2 = im2Display.copy()

    tris1 = []
    tris2 = []
    for i in range(0, len(dt)):
        tri1 = []
        tri2 = []
        for j in range(0, 3):
            tri1.append(hull1[dt[i][j]])
            tri2.append(hull2[dt[i][j]])

        tris1.append(tri1)
        tris2.append(tri2)

    cv2.polylines(imTemp1,np.array(tris1),True,(0,0,255),2);
    cv2.polylines(imTemp2,np.array(tris2),True,(0,0,255),2);
    for i in range(0, len(tris1)):
        fbc.warpTriangle(img1, img1Warped, tris1[i], tris2[i])
    output = cv2.seamlessClone(np.uint8(img1Warped[:,:,::-1]), img2, mask, center,cv2.NORMAL_CLONE)
    ### Default scaling to 25 percent
    scale_percent = 25
    width = int(output.shape[1] * scale_percent / 100)
    height = int(output.shape[0] * scale_percent / 100)

    # dsize
    dsize = (width, height)

    # resize image
    output = cv2.resize(output, dsize)

    return output

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

        #### Randomly select a body 
        img_host,img_class = select_random_body()

        swapped_img = execute_face_swap(aligned_face, img_host)

        ### We will return a base64 encoded JPG file so will set the content type etc accordingly
        resultant_image = base64.b64encode(cv2.imencode('.jpg', swapped_img)[1])
        #multipart_encoded_reply =  MultipartEncoder({'field2': ('filename', resultant_image, 'image/jpg')})
        
        return { 
            "statusCode": 200, 
            "headers":{ 
                'Content-Type': 'application/json',#'image/jpg', #multipart_encoded_reply.content_type , 
                'Access-control-Allow-origin': '*', 
                'Access-control-Allow-credentials' : True
            },
            "body": json.dumps({'file':filename.replace('"',''), 'facecount': len(faceCount), 'norseclass': img_class,'image':resultant_image.decode()})#multipart_encoded_reply.to_string()
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
