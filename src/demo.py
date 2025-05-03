from infer import picture_main,video_main, webcam_main

# I have to do this since this calls a GPU process
if __name__ == '__main__':
    picture_main("./demo_2.jpg")
    picture_main("./demo_3.jpg")
    # video_main("./demo.mp4")
    webcam_main()
