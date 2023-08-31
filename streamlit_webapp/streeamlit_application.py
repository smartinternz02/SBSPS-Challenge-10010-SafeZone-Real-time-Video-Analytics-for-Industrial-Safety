import streamlit as st
from PIL import Image
from object_detection_video_image import *
import tempfile
def main():
    st.title("OBJECT DETECTION USING YOLO-NAS CUSTOM DATASET")
    st.sidebar.title("Settings")
    st.sidebar.subheader("parameters")
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded = "true"] > div:first-child{
            width:300px;
        }
        
        </style>
        """,
        unsafe_allow_html=True,
    )


    app_mode = st.sidebar.selectbox('Choose the App Mode', ['About App','Run on Image','Run on Video'])
    if app_mode == 'About App':
        st.markdown("ravi is always great as usual")
        st.markdown('In This Project I am using ***Yolo-Nas*** with custom dataset after training the model for 11 hours to do Object Detection on Images and Videos and we are using  Stream Lit as an webapp to create a GUI ')
        st.markdown(
            """
            <style>
            [data-testid="stSidebar"][aria-expanded = "true"] > div:first-child{
                width:300px;
            }
    
            </style>
            """,
            unsafe_allow_html=True,
        )
    elif app_mode == 'Run on Image':
        st.sidebar.markdown('---')
        confidence = st.sidebar.slider('Confidence', min_value=0.0, max_value=1.0)
        st.markdown(
            """
            <style>
            [data-testid="stSidebar"][aria-expanded = "true"] > div:first-child{
                width:300px;
            }
        
            </style>
            """,
            unsafe_allow_html=True,
        )
        img_file_buffer = st.sidebar.file_uploader("UPLOAD AN IMAGE", type=["jpg", "jpeg", "png"])
        demo_img = 'D:/PROJECTS/ibm_hackathon/project_industry/newnew_data/data/p1.png'
        if img_file_buffer is not None:
            img = cv2.imdecode(np.fromstring(img_file_buffer.read(), np.uint8), 1)
            image = np.array(Image.open(img_file_buffer))
        else:
            img = cv2.imread(demo_img)
            image = np.array(Image.open(demo_img))
        st.sidebar.text('ORIGINAL IMAGE')
        st.sidebar.image(image)
        image_detection(image,confidence,st)

    elif app_mode == 'Run on Video':
        st.markdown(
            """
            <style>
            [data-testid="stSidebar"][aria-expanded = "true"] > div:first-child{
                width:300px;
            }

            </style>
            """,
            unsafe_allow_html=True,
        )
        st.sidebar.markdown('----*****-----')
        use_webcam = st.sidebar.checkbox('USE WEBCAM')
        st.sidebar.markdown('----')
        confidence = st.sidebar.slider('Confidence', min_value=0.0, max_value=1.0)
        video_file_buffer = st.sidebar.file_uploader("UPLOAD A VIDEO", type = ["mp4","avi","mov","asf"]) # here this line is for adding the video file into the streamlit by the user
        demo_video ="D:/PROJECTS/ibm_hackathon/project_industry/streamlit_webapp/videos/v1.mp4"
        temfile = tempfile.NamedTemporaryFile(suffix='.mp4',delete=False)
        if not video_file_buffer:
            if use_webcam:
                temfile.name=0
            else:
                vide= cv2.VideoCapture(demo_video)
                temfile.name = demo_video
                demo_vid= open(temfile.name, 'rb')
                demo_bytes =demo_vid.read()
                st.sidebar.text('INPUT VIDEO')
                st.sidebar.video(demo_bytes)


        else:
            temfile.write(video_file_buffer.read())
            demo_vid= open(temfile.name, 'rb')
            demo_bytes = demo_vid.read()
            st.sidebar.text('INPUT VIDEO')
            st.sidebar.video(demo_bytes)

        stframe = st.empty()
        st.markdown("<hr/",unsafe_allow_html = True)
        kpi1,kpi2,kpi3 = st.columns(3)
        with kpi1:
            st.markdown("***FRAME RATE***")
            kpi1_text = st.markdown("0")
        with kpi2:
            st.markdown("***WIDTH***")
            kpi2_text=st.markdown("0")

        with kpi3:
            st.markdown("***HEIGHT***")
            kpi3_text=st.markdown("0")
        st.markdown("<hr/",unsafe_allow_html = True)
        video_detection(temfile.name,kpi1_text,kpi2_text,kpi3_text,stframe)







if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass
