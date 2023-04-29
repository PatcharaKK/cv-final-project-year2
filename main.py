import cv2
import numpy as np
from PIL import Image

def get_image_position(image):
	height, width, _ = image.shape
	position = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

	return position
	
face_filter_image_1 = cv2.imread("./asset/face_filter_1.png", -1)
face_filter_image_2 = cv2.imread("./asset/face_filter_2.png", -1)
face_filter_image_3 = cv2.imread("./asset/face_filter_3.png", -1)
face_filter_image_4 = cv2.imread("./asset/face_filter_4.png", -1)

face_filter_info_1 = (face_filter_image_1, get_image_position(face_filter_image_1))
face_filter_info_2 = (face_filter_image_2, get_image_position(face_filter_image_2))
face_filter_info_3 = (face_filter_image_3, get_image_position(face_filter_image_3))
face_filter_info_4 = (face_filter_image_4, get_image_position(face_filter_image_4))

sharpening_kernel = np.float32([
	[1, 4, 6, 4, 1],
	[4, 16, 24, 16, 4],
	[6, 24, -476, 24, 6],
	[4, 16, 24, 16, 4],
	[1, 4, 6, 4, 1]
])
sharpening_kernel = sharpening_kernel * (-1 / 256)

def ease_in_out(t):
	return 2 * t * t if t < 0.5 else 1 - pow(-2 * t + 2, 2) / 2

class GlamourGurusVideo:
	def __init__(self, index):
		video_capture = cv2.VideoCapture(index)
		video_capture_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
		video_capture_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

		self.video_capture = video_capture
		self.video_capture_width = video_capture_width
		self.video_capture_height = video_capture_height

		self.frame = None

		self.frame_width_start = self.frame_width_end = video_capture_width
		self.frame_height_start = self.frame_height_end = video_capture_height
		self.frame_centerx_start = self.frame_centerx_end = video_capture_width / 2
		self.frame_centery_start = self.frame_centery_end = video_capture_height / 2

	def get_faces(self):
		frame = self.frame
		if frame is None: return []

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		face_classifier = f"{cv2.data.haarcascades}haarcascade_frontalface_default.xml"
		face_cascade = cv2.CascadeClassifier(face_classifier)

		faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(200, 200), flags=cv2.CASCADE_SCALE_IMAGE)
		return faces

	def get_filter_image_info(self, index):
		if index == 0:
			return None
		elif index == 1:
			return face_filter_info_1
		elif index == 2:
			return face_filter_info_2
		elif index == 3:
			return face_filter_info_3
		elif index == 4:
			return face_filter_info_4

	def filter_faces(self, faces, filter_index):
		if len(faces) == 0: return

		filter_image_info = self.get_filter_image_info(filter_index)
		if filter_image_info is not None:
			(filter_image, filter_image_position)  = filter_image_info
		
			for face in faces:
				(x, y, w, h) = face
				offsetx = 50
				offsety = 75
				face_position = np.float32([
					[x - offsetx, y - offsety],
					[x + w + offsetx, y - offsety],
					[x + w + offsetx, y + h],
					[x - offsetx, y + h]
				])

				M = cv2.getPerspectiveTransform(filter_image_position, face_position)
				filtered_frame = cv2.warpPerspective(filter_image, M, (self.video_capture_width, self.video_capture_height))

				filtered_frame_image = Image.fromarray(filtered_frame)
				frame_image = Image.fromarray(self.frame)

				frame_image.paste(filtered_frame_image, (0, 0), filtered_frame_image)

				self.frame = np.array(frame_image)
	
	def show_frame(self, t):
		frame = self.frame
		if frame is None: return

		x = ease_in_out(t)

		frame_width = (self.frame_width_end * x) + (self.frame_width_start * (1 - x))
		frame_height = (self.frame_height_end * x) + (self.frame_height_start * (1 - x))
		frame_centerx = (self.frame_centerx_end * x) + (self.frame_centerx_start * (1 - x))
		frame_centery = (self.frame_centery_end * x) + (self.frame_centery_start * (1 - x))

		cropped_frame = cv2.getRectSubPix(frame, (int(frame_width), int(frame_height)), (int(frame_centerx), int(frame_centery)))
		resized_frame = cv2.resize(cropped_frame, (self.video_capture_width, self.video_capture_height))
		sharpened_frame = cv2.filter2D(resized_frame, 0, sharpening_kernel)

		cv2.imshow("webcam", sharpened_frame)
		

def main():
	glamour_gurus_video = GlamourGurusVideo(0)

	video_capture = glamour_gurus_video.video_capture
	video_capture_width = glamour_gurus_video.video_capture_width
	video_capture_height = glamour_gurus_video.video_capture_height
	video_capture_ratio = video_capture_width / video_capture_height

	filter_index = 1
	t = 0

	video_capturing = True
	while video_capturing:
		ret, frame = video_capture.read()

		if ret:
			glamour_gurus_video.frame = frame

			faces = glamour_gurus_video.get_faces()

			if len(faces) > 0:
				top_left_xs = map(lambda face: face[0], faces)
				top_left_ys = map(lambda face: face[1], faces)
				bottom_right_xs = map(lambda face: face[0] + face[2], faces)
				bottom_right_ys = map(lambda face: face[1] + face[3], faces)

				min_top_left_x = min(top_left_xs)
				min_top_left_y = min(top_left_ys)
				max_bottom_right_x = max(bottom_right_xs)
				max_bottom_right_y = max(bottom_right_ys)

				width = max_bottom_right_x - min_top_left_x
				height = max_bottom_right_y - min_top_left_y
				ratio = width / height

				center_x = min_top_left_x + (width / 2)
				center_y = min_top_left_y + (height / 2) - 50

				if ratio > video_capture_ratio:
					height = width / video_capture_ratio
				elif ratio < video_capture_ratio:
					width = height * video_capture_ratio

				width = width * 1.75
				height = height * 1.75

				if center_x - (width / 2) < 0:
					center_x = width / 2
				elif center_x + (width / 2) > video_capture_width:
					center_x = video_capture_width - (width / 2)

				if center_y - (height / 2) < 0:
					center_y = height / 2
				elif center_y + (height / 2) > video_capture_height:
					center_y = video_capture_height - (height / 2)

				if t == 0:
					glamour_gurus_video.frame_width_end = width
					glamour_gurus_video.frame_height_end = height
					glamour_gurus_video.frame_centerx_end = center_x
					glamour_gurus_video.frame_centery_end = center_y
			else:
				if t == 0:
					glamour_gurus_video.frame_width_end = video_capture_width
					glamour_gurus_video.frame_height_end = video_capture_height
					glamour_gurus_video.frame_centerx_end = video_capture_width / 2
					glamour_gurus_video.frame_centery_end = video_capture_height / 2

			glamour_gurus_video.filter_faces(faces=faces, filter_index=filter_index)
			glamour_gurus_video.show_frame(t)

			t += 0.075
			if t > 1:
				t = 0

				glamour_gurus_video.frame_width_start = glamour_gurus_video.frame_width_end
				glamour_gurus_video.frame_height_start = glamour_gurus_video.frame_height_end
				glamour_gurus_video.frame_centerx_start = glamour_gurus_video.frame_centerx_end
				glamour_gurus_video.frame_centery_start = glamour_gurus_video.frame_centery_end

		key_hex = cv2.waitKey(1) & 0xFF

		if key_hex == ord("q"):
			video_capturing = False
		elif key_hex == ord("0"):
			filter_index = 0
		elif key_hex == ord("1"):
			filter_index = 1
		elif key_hex == ord("2"):
			filter_index = 2
		elif key_hex == ord("3"):
			filter_index = 3
		elif key_hex == ord("4"):
			filter_index = 4

	video_capture.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()