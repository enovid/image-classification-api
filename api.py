import responder
import aiohttp
from fastai.vision import *

# Initialize app
api = responder.API()

# Setup model
path = Path('.')
learner = load_learner('.')
defaults.device = torch.device('cpu')

# Show basic instructions on homepage
@api.route("/")
def index(req, resp):
    resp.text = "Usage: /classify-url?url=<image_url>"


async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()


@api.route("/classify-url")
async def classify_url(req, resp):
    bytes = await get_bytes(req.params["url"])
    img = open_image(BytesIO(bytes))
    _, _, losses = learner.predict(img)
    prediction_text = sorted(zip(learner.data.classes, map(float, losses)),  
                             key=lambda p: p[1], reverse=True)
    resp.text = 'predictions:\n'
    for label, likelihood in prediction_text:
        resp.text += f'\t {label} {round(likelihood * 100, 1)}% \n'

if __name__ == '__main__':
    api.run()
