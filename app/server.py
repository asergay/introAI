import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

#google drive and dropbox can break the files when downloading so we are using a service made by CERN 
# to allow researchers to upload datsets and models and other files

#you can make an account on zendo https://zenodo.org using your github account
export_file_url = 'https://www.dropbox.com/s/v2v5osbpk4m6i4s/driver?dl=0'
#here is the name of the file
export_file_name = 'driver'

#we are using the distracted driver model so we set up its class names.
#make sure the ordering is the same as the order of labels you used in training the model
classes = ['c0: normal driving',
'c1: texting - right',
'c2: talking on the phone - right',
'c3: texting - left',
'c4: talking on the phone - left',
'c5: operating the radio',
'c6: drinking',
'c7: reaching behind',
'c8: hair and makeup',
'c9: talking to passenger']

#we set up the path, we use the parent attribute here as this particular file(server.py) is inside a folder called app
# and we want to access the folder right above this one
path = Path(__file__).parent

#starlette is a wrapper around Flask, just like FastAI is around PyTorch and Keras is around Tensorflow
#it is the reccomended tool used by FastAI
#we create an app object using the main starlette class
app = Starlette()
#this line allows cross-origin access to other servers
#we don't cover this in this course but suffice to say your page can not be dynamic without it
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
#we set up the directory for our static files
#static files are things like images, css and javascript
#these are files that the server code does not change but just displays
app.mount('/static', StaticFiles(directory='app/static'))

#we define a function to download a file
#async tells the function that it is asynchronous meaning it can start processing while letting other functions continue
#when called, it needs a corrosponding await which will wait for this function to finish working before procedding 
#we need async and await when working with communcation over the internet
async def download_file(url, dest):
    #if the destination file already exists, we exit otherwise we download it
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            #here we see an example of await
            #the client session will call the url and the line after data = won't run until it gets a response and 
            #reads it into the data variable
            data = await response.read()
            #now we open the fine we defined in dest and write the contents of what we downloaded and read in data
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    #we download the file using the download_file function we define above
    #notice the use of await which means the code won't proceed until there is a file in the destination
    #and said destination is our path (the folder in which we have the app folder) which should have a 
    #file with the name we set for our export_file_name
    await download_file(export_file_url, path / export_file_name)
    #we use try and catch to load the model into a FastAI learner object 
    # to account for the possibility the downloaded model could be from 
    # an older version and may not work
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise

#these three lines should be read in reverse order as the first two are just setup for the third
#in order to understand them, read the comments in the numbered order

# 3.0 here we set up the event loop that will run until the task defined below is complete
loop = asyncio.get_event_loop()
# 2.0 here we define the task to run, in this case, the only task as the setup_learner function
tasks = [asyncio.ensure_future(setup_learner())]
# 1.0 as the learner object is coming from an asyncronous function
#we set a loop to run until the task is complete and we either get an error message or the learner object
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

#this is our index page which runs when we go to the server in our browser
#app.route is used in the same way as in Flask
@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())

#the function that runs the prediction and returns the result
#we use a POST method here as we need to call this route with some form data
#in index.html, if you look at the code for the analyze button, you will see how this function is called
@app.route('/analyze', methods=['POST'])
async def analyze(request):
    #gets the image data from our form defined in index.html
    img_data = await request.form()
    #converts the data to bytes
    img_bytes = await (img_data['file'].read())
    #converts the bytes to an image object
    img = open_image(BytesIO(img_bytes))
    #runs prediction function on our image
    prediction = learn.predict(img)[0]
    #and finally returns a response as JSON
    return JSONResponse({'result': str(prediction)})


if __name__ == '__main__':
    #when we want to run this file locally, we call it with the 'serve' parameter 
    if 'serve' in sys.argv:
        #uvicorn is the server we use and we set it to run the app object with the given parameters
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
