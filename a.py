import matplotlib.pyplot as plt 
import numpy as np
import cv2
import random

def imageStitching(img1, img2):
	sift = cv2.xfeatures2d.SIFT_create(800)
	kp1, des1 = sift.detectAndCompute(img1, None)
	kp2, des2 = sift.detectAndCompute(img2, None)

	cv2.drawKeypoints(img1, kp1, img1)
	cv2.drawKeypoints(img2, kp2, img2)

	plt.imsave("set1_featurePoints.jpg", np.hstack([img1, img2]))

	kp1Array = []
	for each in kp1:
		kp1Array.append([each.pt[0], each.pt[1], each.size, each.angle])

	kp2Array = []
	for each in kp2:
		kp2Array.append([each.pt[0], each.pt[1], each.size, each.angle])

	dist = np.array([[i, j, np.linalg.norm(each-every)] for i, each in enumerate(des1) for j, every in enumerate(des2)])

	min_dist = min(dist[:, -1])

	matches = [cv2.DMatch(int(each[0]), int(each[1]), each[2]) for each in dist if each[2] < min_dist*3]

	matches.sort(key=lambda u: u.distance, reverse=True)

	img3 = cv2.hconcat([img1, img2])

	draw_params = dict(matchColor = (255, 255, 0), singlePointColor = None, flags = 2)
	cv2.drawMatches(img1, kp1, img2, kp2, matches, img3, **draw_params)

	rand = [random.randint(0, len(matches)) for i in range(3)]

	points1 = [kp1Array[matches[i].queryIdx][0:2] for i in rand]
	points2 = [kp2Array[matches[i].trainIdx][0:2] for i in rand]

	plt.imsave("set1_matches.jpg", img3)
	plt.show()

	rows, cols = img1.shape[0], img1.shape[1]+img2.shape[1]

	A = np.array([
			[points2[0][0], points2[0][1], 1, 0, 0, 0],
			[0, 0, 0, points2[0][0], points2[0][1], 1],
			[points2[1][0], points2[1][1], 1, 0, 0, 0 ],
			[0, 0, 0, points2[1][0], points2[1][1], 1],
			[points2[2][0], points2[2][1], 1, 0, 0, 0],
			[0, 0, 0, points2[2][0], points2[2][1], 1]
		])

	B = np.array([
			points1[0][0], points1[0][1], points1[1][0], points1[1][1], points1[2][0], points1[2][1]
		])

	A_inverse = np.linalg.pinv(A)

	T = np.reshape(np.dot(A_inverse, B), (2, 3))

	T_image = cv2.warpAffine(img2, T, (cols, rows))

	plt.imshow(T_image)
	plt.show()

	new_image = np.zeros_like(img3)

	for i in range(new_image.shape[0]):
		for j in range(new_image.shape[1]):
			if T_image[i, j].all() == 0:
				if i < img1.shape[0] and j < img1.shape[1]:
					new_image[i, j] = img1[i, j]
			else:
				new_image[i, j] = T_image[i, j]

	plt.imsave("set1_panorama.jpg", new_image)
	plt.show()

	return new_image


def panorama(images):
	old = plt.imread(images[0])
	new = None
	for i in range(1, len(images)):
		new = plt.imread(images[i])
		old = imageStitching(old, new)
	return old


img = panorama(["image1.jpg", "image2.jpg"])


