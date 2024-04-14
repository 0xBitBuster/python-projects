from CardDetector import CardDetector

if __name__ == "__main__":
    video_source = 0
    card_detector = CardDetector(video_source)
    card_detector.process_video()
