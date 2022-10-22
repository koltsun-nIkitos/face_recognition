import os
import pickle
import sys
import face_recognition
import cv2


def train_model_by_img(name):

    if not os.path.exists("dataset"):
        print("[ERROR] нет файлов в 'dataset'")
        sys.exit()

    known_encodings = []
    images = os.listdir("dataset")

    # print(images)

    for(i, image) in enumerate(images):
        print(f"[+] процесс {i + 1}/{len(images)}")
        # print(image)

        face_img = face_recognition.load_image_file(f"dataset/{image}")
        face_enc = face_recognition.face_encodings(face_img)[0]

        # print(face_enc)

        if len(known_encodings) == 0:
            known_encodings.append(face_enc)
        else:
            for item in range(0, len(known_encodings)):
                result = face_recognition.compare_faces([face_enc], known_encodings[item])
                # print(result)

                if result[0]:
                    known_encodings.append(face_enc)
                    # print("Нашел!!")
                    break
                else:
                    # print("Другой человек!")
                    break

    # print(known_encodings)
    # print(f"Длина {len(known_encodings)}")

    data = {
        "name": name,
        "encodings": known_encodings
    }

    with open(f"{name}_encodings.pickle", "wb") as file:
        file.write(pickle.dumps(data))

    return f"[INFO] Файл {name}_encodings.pickle успешно создан!"


def take_screenshot_from_video():
    cap = cv2.VideoCapture("video.mp4")
    count = 0

    if not os.path.exists("dataset_from_video"):
        os.mkdir("dataset_from_video")

    while True:
        ret, frame = cap.read()
        fps = cap.get(cv2.CAP_PROP_FPS)
        multiplier = fps * 3
        #print(fps)
        
        if ret:
            fram_id = int(round(cap.get(1)))
            #print(fram_id)
            cv2.imshow("frame", frame)
            k = cv2.waitKey(20)

            if fram_id % multiplier == 0:
                cv2.imwrite(f"dataset_from_video/{count}screen.jpg", frame)
                print(f"Сделал скриншот! {count}")
                count += 1

            if k == ord(" "):
                cv2.imwrite(f"dataset_from_video/{count}_extra_screen.jpg", frame)
                print(f"Сделал моментальный скриншот! {count}")
                count += 1
            
            if k == ord("q"):
                print("Q нажата, закрытие программы")
                break
        else:
            print("[ERROR] не могу получить кадр!")
            break

    cap.release()
    cv2.destroyAllWindows()
        



def main():
    print(train_model_by_img("Nikita_Koltsun"))
    #take_screenshot_from_video()

if __name__ == '__main__':
    main()