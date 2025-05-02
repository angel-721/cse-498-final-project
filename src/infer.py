from ultralytics import YOLO
import cv2
import numpy as np
import math

classes = ['10c', '10d', '10h', '10s', '2c', '2d', '2h', '2s', '3c', '3d', '3h', '3s', '4c', '4d', '4h', '4s', '5c', '5d', '5h', '5s', '6c', '6d', '6h', '6s', '7c', '7d', '7h', '7s', '8c', '8d', '8h', '8s', '9c', '9d', '9h', '9s', 'Ac', 'Ad', 'Ah', 'As', 'Jc', 'Jd', 'Jh', 'Js', 'Kc', 'Kd', 'Kh', 'Ks', 'Qc', 'Qd', 'Qh', 'Qs']

total_cards = 0
MIN_DISTANCE = 300
hand_count = 0


# a hand will have a key that's a set of the cards in the hand and value of the
# total of the hand
hands = {}
used_cards = set()

model = YOLO("./models/yolo11n-cards-60-epochs.pt")

# Open the default camera
cam = cv2.VideoCapture(0)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

def card_id_to_value(card_id):
    # convert face cards to 10
    if card_id in ['10c', '10d', '10h', '10s', 'Jc', 'Jd', 'Jh', 'Js', 'Qc',
                   'Qd', 'Qh', 'Qs', 'Kc', 'Kd', 'Kh', 'Ks']:

        return 10

    elif card_id in ['Ac', 'Ad', 'Ah', 'As']:
        return 11

    else:
        # Remove the suit and convert to int
        return int(card_id[:-1])

def middle_of_box(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def euclidean_distance(box1, box2):
    x1, y1 = middle_of_box(box1)
    x2, y2 = middle_of_box(box2)
    return math.sqrt(math.pow(x2- x1, 2) + math.pow(y2 - y1, 2))

def find_nearest_card(card, card_boxes, class_ids, card_id):
    nearest_card = None
    min_distance = float('inf')
    nearest_index = -1

    for i, box in enumerate(card_boxes):
        # skip if the card has the same id
        if classes[class_ids[i]] == card_id:
            continue
        if classes[class_ids[i]] in used_cards:
            continue

        distance = euclidean_distance(card, box)
        if distance < MIN_DISTANCE and distance < min_distance:
            min_distance = distance
            nearest_card = box
            nearest_index = i

    return nearest_card, nearest_index

def generate_hands(boxes, class_ids):
    global hands
    global total_cards
    global used_cards
    global hand_count

    # Reset counters
    total_cards = 0
    hands = {}
    hand_count = 0
    used_cards = set()

    # First pass: create hands of 2 cards
    for i, box in enumerate(boxes):
        # Skip if this card is already in a hand
        if classes[class_ids[i]] in used_cards:
            continue

        # Find nearest card to this one
        nearest_card, nearest_idx = find_nearest_card(box,
                                                      boxes,
                                                      class_ids,classes[class_ids[i]])

        if nearest_card is not None and nearest_idx >= 0:
            # Create a new hand with these two cards
            card_id = classes[class_ids[i]]
            nearest_card_id = classes[class_ids[nearest_idx]]

            # Create a new hand
            hand = [card_id, nearest_card_id]
            hand_key = frozenset(hand)
            print(f"Creating hand with cards: {hand}")
            print(f"Hand key: {hand_key}")


            hands[hand_key] = card_id_to_value(card_id) + card_id_to_value(nearest_card_id)

            # Mark both cards as used
            used_cards.add(card_id)
            used_cards.add(nearest_card_id)


    # Second pass: any cards not in a hand become single-card hands
    for i, box in enumerate(boxes):
        if classes[class_ids[i]] in used_cards:
            card_id = classes[class_ids[i]]
            hand_key = frozenset([card_id])
            hands[hand_key] = card_id_to_value(card_id)
            # Mark the card as used
            used_cards.add(card_id)
            print(f"Created single card hand with {card_id}")

    # Calculate total cards and hands
    total_cards = sum(len(hand) for hand in hands.keys())
    hand_count = len(hands)

def draw_card_data(annotated_frame):
    cv2.putText(annotated_frame, f'Total Cards: {total_cards}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.putText(annotated_frame, f'Total Hands: {hand_count}', (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # draw hands
    for hand_key, hand_value in hands.items():
        # Convert the frozenset back to a list for display
        hand_list = list(hand_key)
        hand_str = ', '.join(hand_list)
        cv2.putText(annotated_frame, f'Hand: {hand_str} Value: {hand_value}', (10, 70 + 20 * list(hands.keys()).index(hand_key)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

def webcam_main():
    global total_cards
    global hands
    global hand_count

    while True:
        _, frame = cam.read()

        # # Resize the frame to the input size of the model
        # resized_frame = cv2.resize(frame, (640, 640))
        results = model.track(source=frame, conf=0.5, iou=0.5)
        annotated_frame = results[0].plot()

        generate_hands(results[0].boxes.xyxy.cpu().numpy(),
                       results[0].boxes.cls.cpu().numpy().astype(int))
        print("hands: ", hands)

        draw_card_data(annotated_frame)

        cv2.imshow('Camera', annotated_frame)


        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

def picture_main(image_path):
    global total_cards

    # Read the image
    image = cv2.imread(image_path)
    # Resize the image to the input size of the model
    resized_image = cv2.resize(image, (640, 640))

    results = model.track(source=resized_image, conf=0.5, iou=0.5)

    annotated_image = results[0].plot()
    generate_hands(results[0].boxes.xyxy.cpu().numpy(),
                   results[0].boxes.cls.cpu().numpy().astype(int))

    draw_card_data(annotated_image)

    # Display the image with bounding boxes
    cv2.imshow('Annotated Image', annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
    # webcam_main()
    picture_main("./test2.jpg")
