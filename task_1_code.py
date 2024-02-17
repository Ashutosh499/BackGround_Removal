import cv2
import rembg

def draw_outline(image, m):
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 2)  

    return img

def main():
    input_image_path = input("Enter the path to the input image: ")
    while True:
        image = cv2.imread(input_image_path)
        if image is None:
            print("Enput Path is not correct!")
            break

        image = cv2.resize(image, (500, 500))
        cv2.namedWindow("ROI Selection", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("ROI Selection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        roi = cv2.selectROI("ROI Selection", image)
        cv2.destroyAllWindows()

        a, b, c, d = roi
        roi_image = image[b:b+d, a:a+c]

        result_image = rembg.remove(roi_image)
        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGRA2BGR)

        gray = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        result = draw_outline(roi_image, mask)

        image[b:b+d, a:a+c] = result

        cv2.namedWindow("Output Image", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Output Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Output Image", image)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('c'):
            break
        elif key == ord('q'):
            cv2.destroyAllWindows()
            continue

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
