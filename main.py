from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO
from fastapi.responses import StreamingResponse

import torch
import utils
from PIL import Image
import torch.nn as nn
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage, Normalize, Resize

from model.rpnet import Net

app = FastAPI()

origins = ["*"]  

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_model():
    global net
    net = Net(13)
    net.load_state_dict(torch.load("./pretrained/I-HAZE_O-HAZE.pth")['state_dict'])
    # net.load_state_dict(torch.load("./pretrained/10.pth")['state_dict'])
    net.eval()
    net = nn.DataParallel(net, device_ids=[0, 1, 2, 3])
    print("==================== Model loaded ==========================")


load_model()

@app.post("/dehaze")
async def upload_file(file: UploadFile):
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
        return {"error": "Only image files (jpg, jpeg, png, gif, bmp) are allowed."}

    image_bytes = await file.read()

    try:
        im = Image.open(BytesIO(image_bytes))
    except Exception as e:
        return {"error": f"Error converting image: {str(e)}"}
    # You can perform any processing on pil_image here if needed


    h, w = im.size
    print(h, w)
    im = ToTensor()(im)
    im = Variable(im).view(1, -1, w, h)
    # im = im
    with torch.no_grad():
        im = net(im)
    im = torch.clamp(im, 0., 1.)
    im = im.cpu()
    im = im.data[0]
    im = ToPILImage()(im)

    # Convert the PIL Image back to bytes
    img_byte_array = BytesIO()
    im.save(img_byte_array, format="PNG")
    img_byte_array = img_byte_array.getvalue()

    # Return the PIL Image as a response
    return StreamingResponse(BytesIO(img_byte_array), media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
