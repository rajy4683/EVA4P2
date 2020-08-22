try: 
    import unzip_requirements
    #from imagenet_labels import  imagenet_labels
except ImportError: 
    print("Importing unzip_requirements  failed")
    pass 
import torch 
import torchvision 
import torchvision.transforms as transforms 
from  PIL import Image
import boto3
import os
import io 
import json 
import base64 
from requests_toolbelt.multipart import decoder

#from .models.inception_resnet_v1 import InceptionResnetV1
from detect_face import extract_face
from mtcnn import MTCNN, PNet, RNet, ONet, prewhiten, fixed_image_standardization

print( "Import End....")

S3_BUCKET = os.environ['S3_BUCKET'] if 'S3_BUCKET' in os.environ else 'rekog-eva4s1'
MODEL_PATH = os.environ['MODEL_PATH'] if 'MODEL_PATH' in os.environ else 'resnet_custom_facedetect_2.pt'

s3 = boto3.client('s3')
try: 
    if os.path.isfile( MODEL_PATH) != True:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_PATH)
        print( "Creating Bytestream") 
        bytestream = io.BytesIO(obj['Body'].read() ) 
        print("Loading Model")
        model = torch.jit.load(bytestream) 
        model.eval()
        mtcnn = MTCNN(
                image_size=160, margin=0, min_face_size=20,
                thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
                device='cpu'
        )
        print( "Model Loaded")
except Exception as e:
    print("Base Model Loading failed!",repr(e))
    raise(e)

#class_names=['Large QuadCopters', 'Flying Birds', 'Winged Drones',
#               'Small QuadCopters']
class_name=['AR_Rehman', 'David_Goggins', 'Gustav_Mahler', 'Janis_Joplin', 'Jim_Morrison', 'Rich_Froning', 'Tia_Clair_Toomey']

def transform_image( image_bytes ) :
    try:
        #transformations = transforms.Compose([
        #                #transforms.Resize(224) ,
        #                #transforms.CenterCrop(224),
        #                np.float32,
        #                transforms.ToTensor(),
        #                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        transformations = transforms.Compose([
	    np.float32,
	    transforms.ToTensor(),
	    fixed_image_standardization
	])
        image = Image.open(io.BytesIO(image_bytes))
        print("Transformations successful!")
        return transformations(image).unsqueeze(0)
    except Exception as e:
        print("Exception during transform_image", repr(e))
        raise(e)

def extract_align_faces(image_bytes):
    img = np.array(Image.open(io.BytesIO(image_bytes)).convert('RGB'))
    face_tensor,probs = mtcnn(mtcnn(img, return_prob=True))
    return face_tensor,probs

def get_prediction( tensor ) : 
    #tensor = transform_image( image_bytes = image_bytes )   
    return model(tensor.unsqueeze(0)).argmax().item() 

def classify_image(event, context) : 
    try: 
        if(event['httpMethod'] == 'OPTIONS'):
            return {
                "statusCode": 200,
                "headers":{
                    'Access-control-Allow-origin': '*',
                    'Access-control-Allow-credentials' : True,
                    'Access-Control-Allow-Methods': 'POST, GET, OPTIONS, DELETE',
                    'Access-Control-Allow-Headers': '*',
                },
                "body":json.dumps({'foo':'bar'})
            }
        content_type_header = event['headers']['content-type'] 
        #print(event['body'])
        body = base64.b64decode(event["body"]) 
        print( 'BODY LOADED' )
        picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
        face_tensor,probs = extract_align_faces(image_bytes = picture.content)
        prediction =  get_prediction(image_bytes = picture.content) 
        print("Get Prediction returned:",prediction, type(prediction), class_names[int(prediction)])
        predicted_val = class_names[int(prediction)]
        predicted_str = f'{predicted_val}'

        content_disp = picture.headers[b'Content-Disposition'].decode().split(';')
        filename = content_disp[1].split('=')[1]
        if len(filename) < 4:
            filename = content_disp[2].split('=')[1]
        
        return { 
            "statusCode": 200, 
            "headers":{ 
                'Content-Type': 'application/json' , 
                'Access-control-Allow-origin': '*', 
                'Access-control-Allow-credentials' : True
            },
            "body":json.dumps({'file':filename.replace('"',''), 'predicted': predicted_str, 'class':int(prediction)})
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
            "body": json .dumps({"error": repr(e)})
        }
