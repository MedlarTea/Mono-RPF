#! /usr/bin/env python3
import numpy as np
import sys
import os
print(os.getcwd())
print("Mono Following Node Python:", sys.executable)
# print(sys.path)
import ros_numpy
import cv2
import rospy
import os.path as osp

# My class
from tracklet import Tracklet
from states.initial_state import InitialState
from descriminator import Descriminator
# standard ROS message
from sensor_msgs.msg import Image
# import tf

# my ROS message
from spencer_tracking_msgs.msg import TrackedPerson, TrackedPersons, TargetPerson, Box
from mono_tracking.msg import TrackArray
from mono_tracking.msg import Track
from mono_following.msg import Target
import message_filters
# from mono_following.msg import Box


class MonoFollowing:
    def __init__(self):
        ### Some parameters for debug and testing ###
        self.evaluate = rospy.get_param("~evaluate")
        # self.SAVE_TRACKS = False
        # self.SAVE_IMAGES = False
        # self.IMAGE_PATH = "/home/jing/Data/Dataset/FOLLOWING/ccf_failure_dataset_masaic/ccf_failure_dataset/room433/debug/grr_slt"
        if self.evaluate:
            self.STORE_DIR = rospy.get_param("~track_store_dir")
            self.result_fname = osp.join(self.STORE_DIR, "Visible-Parts-MPF.txt")
            print("Store result to: {}".format(self.result_fname))
            self.result_fp = open(self.result_fname, "w")
            self.num = 0

        ### Our parameters ###
        self.state = InitialState()
        self.descriminator = Descriminator()

        ### Our necessary topics ###
        # listen to the transform tree
        # self.tf_listener = tf.TransformListener()

        imageTopic = "/camera/color/image_raw"
        self.imageSubscriber = message_filters.Subscriber(imageTopic, Image)
        trackedPersonsTopic = "/mono_tracking/tracks"
        self.tracks_sub = message_filters.Subscriber(trackedPersonsTopic, TrackedPersons)
        tss = message_filters.ApproximateTimeSynchronizer([self.tracks_sub, self.imageSubscriber], queue_size=20, slop=0.8)
        tss.registerCallback(self.callback)

        # publish image for visualization
        self.image_pub = rospy.Publisher("/mono_following/vis_image", Image, queue_size=1)
        # publish image patches for visualization
        self.patches_pub = rospy.Publisher("/mono_following/patches", Image, queue_size=1)
        # publish target information
        self.target_pub = rospy.Publisher("/mono_following/target", TargetPerson, queue_size=1)

        ### Node is already set up ###
        rospy.loginfo("Mono Following Node is Ready!")
        rospy.spin()
    
    def callback(self, tracks_msg, image_msg):
        print("REID CALLBACK")
        # get messages
        self.tracks = {}
        if image_msg.encoding == "rgb8":
            image = ros_numpy.numpify(image_msg)
        elif image_msg.encoding == "bgr8":
            image = ros_numpy.numpify(image_msg)[:,:,[2,1,0]]  # change to rgb

        for track in tracks_msg.tracks:
            if track.pose.pose.position.x == 0 or track.is_matched == False:
                continue
            self.tracks[track.track_id] = Tracklet(tracks_msg.header, track, image)

        # extract features
        self.descriminator.extractFeatures(self.tracks)
        
        # update the state
        # print(self.state.state_name())
        next_state = self.state.update(self.descriminator, self.tracks)
        if next_state is not self.state:
            self.state = next_state
        
        # publish target information
        target = TargetPerson()
        target.header = tracks_msg.header
        target_id = self.state.target()
        
        for idx in self.tracks.keys():
            if idx == target_id:
                target.track_id = target_id
                target.bounding_box = self.tracks[idx].bounding_box if self.tracks[idx].target_confidence != None else Box()
                target.target_confidence = self.tracks[idx].target_confidence if self.tracks[idx].target_confidence != None else 0.0
                target.pose = self.tracks[idx].person_poseWithCovariance
                target.twist = self.tracks[idx].twist
                target.is_matched = True

        self.target_pub.publish(target)

        # save result
        if self.evaluate:
            self.save_result(target.bounding_box)
        
        # publish image containing the identification result
        if self.image_pub.get_num_connections():
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image_bgr = self.visualize(image_bgr, self.tracks)
            image_msg = ros_numpy.msgify(Image, image_bgr, encoding = "bgr8")
            self.image_pub.publish(image_msg)

        # publish target patch to see whether is OK
        if self.patches_pub.get_num_connections():
            image = self.visualize_patches(self.tracks)
            if image is not None:
                # cv2.imwrite("./test.jpg", image)
                image_msg = ros_numpy.msgify(Image, image, encoding = "rgb8")
                self.patches_pub.publish(image_msg)

        # Save tracks information, for debug and analysis

    def visualize(self, original_image, tracks):
        # print("VISUALIZE")
        image = original_image.copy()
        for idx in tracks.keys():
            if tracks[idx].target_confidence == None:
                continue
            image = cv2.putText(original_image, self.state.state_name(), (20, 60), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
            if idx == self.state.target():
                image = cv2.rectangle(image, (self.tracks[idx].region[0],self.tracks[idx].region[1]), (self.tracks[idx].region[2],self.tracks[idx].region[3]), (0,255,0), 3)
                image = cv2.putText(image, "id:{:d}".format(idx), ((int((self.tracks[idx].region[0]+self.tracks[idx].region[2])/2-5), int((self.tracks[idx].region[1]+self.tracks[idx].region[3])/2-5))), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 1)
                # image = cv2.putText(image, "score:{:.3f}".format(self.tracks[idx].target_confidence), ((int((self.tracks[idx].region[0]+self.tracks[idx].region[2])/2-35), int((self.tracks[idx].region[1]+self.tracks[idx].region[3])/2-35))), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 1)
            else:
                image = cv2.rectangle(image, (self.tracks[idx].region[0],self.tracks[idx].region[1]), (self.tracks[idx].region[2],self.tracks[idx].region[3]), (0,0,255), 3)
                image = cv2.putText(image, "id:{:d}".format(idx), ((int((self.tracks[idx].region[0]+self.tracks[idx].region[2])/2-5), int((self.tracks[idx].region[1]+self.tracks[idx].region[3])/2-5))), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,255), 1)
                # image = cv2.putText(image, "score:{:.3f}".format(self.tracks[idx].target_confidence), ((int((self.tracks[idx].region[0]+self.tracks[idx].region[2])/2-35), int((self.tracks[idx].region[1]+self.tracks[idx].region[3])/2-35))), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,255), 1)

        return image
    
    def saveTrack(self, track_id, track, store_path):
        print("SAVE TRACKS")

    def visualize_patches(self, tracks):
        target_id = self.state.target()
        if target_id in tracks.keys() and self.tracks[target_id].image_patch is not None:
            return self.tracks[target_id].image_patch
        else:
            return None
    
    def save_result(self, bbox:Box):
        """Save target bbox result for evaluation, this is for icvs dataset.
        Input:
            bbox messgae: [x1, y1, w, h]
        Write:
            bbox as [x1, y1, w, h]
        """
        self.result_fp.write("{} {} {} {} {}\n".format(self.num, int(bbox.x), int(bbox.y), int(bbox.w), int(bbox.h)))
        self.num += 1

if __name__ == "__main__":
    rospy.init_node('mono_following', anonymous=True)
    mono_following = MonoFollowing()

