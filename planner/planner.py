import openai
import os
import numpy as np
import open3d as o3d
from copy import deepcopy
from PIL import Image
import re
import cv2
from utils import load_prompt, encode_image
from utils import farthest_point_sampling, fps_rad_idx, truncate_points

from dynamics.plan import closed_loop_plan
from perception.predictor import GroundingSegmentPredictor
from planner.prompt_retriever import PromptRetriever


class KUDAPlanner:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.top_down_cam = self.config['top_down_cam']
        self.obs_x_center = self.config['obs_x_center']
        self.obs_y_center = self.config['obs_y_center']
        self.obs_x_scale = self.config['obs_x_scale']
        self.obs_y_scale = self.config['obs_y_scale']

        self.tracking_cam = self.config['tracking_cam']
        self.close_loop = self.config['close_loop']
        self.log_dir = self.config['log_dir']
        os.makedirs(self.log_dir, exist_ok=True)

        self.base_prompt = load_prompt(f'{self.config["prompt_fname"]}.txt')
        self.few_shot_prompt_dir = os.path.join('prompts', self.config['few_shot_prompt_dir'])
        if not os.path.exists(self.few_shot_prompt_dir):
            os.makedirs(self.few_shot_prompt_dir)

        self.use_retriever = self.config['use_retriever']
        if self.use_retriever:
            self.retriever = PromptRetriever()

        self.target_pcd = None
        if self.config['target_pcd_file'] is not None:
            self.target_pcd = o3d.io.read_point_cloud(self.config['target_pcd_file'])
            self.target_pcd = np.array(self.target_pcd.points)
            self.target_pcd = self.target_pcd[np.newaxis, ...]

    def _build_few_shot_prompt(self, query, obs, encoded_obs):
        examples = []
        # enumerate all directories in base_dir
        for example_dir in os.listdir(self.few_shot_prompt_dir):
            example_dir = os.path.join(self.few_shot_prompt_dir, example_dir)
            if not os.path.isdir(example_dir):
                continue

            query_file = os.path.join(example_dir, 'query.txt')
            with open(query_file, 'r') as f:
                example_query = f.read().strip()
            if example_query == 'not used':
                continue
            obs_file = os.path.join(example_dir, 'obs.png')
            example_obs = Image.open(obs_file)
            # encoded_example_obs = encode_image(example_obs)
            response_file = os.path.join(example_dir, 'response.txt')
            with open(response_file, 'r') as f:
                response = f.read().strip()

            examples.append((example_query, example_obs, response))

        if self.use_retriever:
            obs = Image.fromarray(obs[..., ::-1])
            examples = self.retriever(query, obs, examples)

        few_shot_messages = []
        for example_query, example_obs, response in examples:
            few_shot_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f'Task instruction: {example_query}'
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{encode_image(example_obs)}"
                            }
                        },
                        {
                            "type": "text",
                            "text": response
                        }
                    ]
                }
            )

        return few_shot_messages

    def _build_prompt(self, annotated_image, instruction):
        assistant = f'Got it. I will give response based on what you give me next.'
        user_query = f'Task instruction: {instruction}'
        encoded_image_top_down = encode_image(annotated_image)

        messages = [
            {"role": "system", "content": "You are a helpful assistant that pays attention to the user's demands and is good at writing python code for object manipulation in a tabletop environment."},
            {"role": "user", "content": self.base_prompt},
            *self._build_few_shot_prompt(instruction, annotated_image, encoded_image_top_down),
            {"role": "assistant", "content": assistant},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_query,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded_image_top_down}"
                        }
                    },
                ]
            }
        ]

        return messages

    def _crop_obs(self, img):
        img = Image.fromarray(img)
        W, H = img.size
        x_min = self.obs_x_center - int(W * self.obs_x_scale / 2)
        x_max = self.obs_x_center + int(W * self.obs_x_scale / 2)
        y_min = self.obs_y_center - int(H * self.obs_y_scale / 2)
        y_max = self.obs_y_center + int(H * self.obs_y_scale / 2)
        img = img.crop((x_min, y_min, x_max, y_max))
        img = img.resize((W, H), Image.Resampling.LANCZOS)
        img = np.array(img)
        return img

    def _apply_cropped_transform(self, pcds, W, H):
        new_pcds = []
        for points in pcds:
            points = np.array(points)
            x_min = self.obs_x_center - int(W * self.obs_x_scale / 2)
            y_min = self.obs_y_center - int(H * self.obs_y_scale / 2)
            points = points - np.array([x_min, y_min])
            points = (points / np.array([self.obs_x_scale, self.obs_y_scale])).astype(int)
            new_pcds.append(points)
        return new_pcds

    def _ground_point(self, point, W, H):
        x_min = self.obs_x_center - int(W * self.obs_x_scale / 2)
        y_min = self.obs_y_center - int(H * self.obs_y_scale / 2)
        x_origin = x_min + int(point[0] * self.obs_x_scale)
        y_origin = y_min + int(point[1] * self.obs_y_scale)
        return x_origin, y_origin

    def _view_specification(self, image, object_points, target_specification):
        img = deepcopy(image)
        for index, destination in zip(*target_specification):
            start = object_points[index]
            start_image, end_image = self.env.world_to_viewport([start, destination], self.top_down_cam)
            start_crop_image, end_crop_image = self._apply_cropped_transform([[start_image, end_image]], img.shape[1], img.shape[0])[0]
            # draw an arrow from start to end
            cv2.arrowedLine(img, (int(start_crop_image[0]), int(start_crop_image[1])), (int(end_crop_image[0]), int(end_crop_image[1])), (0, 0, 255), 2)
        return img

    def __call__(self, object, material, instruction, call_depth=0):
        rgbs, _, depths, _ = self.env.get_rgb_depth_pc()
        obs = rgbs[self.top_down_cam]
        H, W = obs.shape[:2]
        cropped_obs = self._crop_obs(obs)
        depth = depths[self.top_down_cam]

        # get the keypoints
        masks = self.env.get_all_masks(cropped_obs, debug=False)
        masks.pop() # remove the background mask
        annotated_points = []
        # keypoint hyperparameters
        num_per_obj = 8
        # Check before experiment
        radius = 200
        # for cubes, use the center to get the annotated points
        if material == 'cube':
            for mask in masks:
                positions = np.argwhere(mask)
                positions = positions[:, [1, 0]]
                keypoint = positions.mean(axis=0)
                annotated_points.append(keypoint)
        else:
            # use fps to get the annotated points
            for mask in masks:
                positions = np.argwhere(mask)
                positions = positions[:, [1, 0]]
                fps_1 = farthest_point_sampling(positions, num_per_obj)
                fps_2, _ = fps_rad_idx(fps_1, radius)
                annotated_points.extend(fps_2)
                # add center for each mask
                center = positions.mean(axis=0)
                annotated_points.append(center)
        # global radius
        global_radius = 40
        annotated_points, _ = fps_rad_idx(np.array(annotated_points), global_radius)

        # get the annotated image
        annotated_image = get_annotated_image(cropped_obs, annotated_points, debug=False, mask=masks)

        # GPT planning
        messages = self._build_prompt(annotated_image, instruction)
        ret = openai.ChatCompletion.create(
            messages=messages,
            temperature=self.config['temperature'],
            model=self.config['model'],
            max_tokens=self.config['max_tokens']
        )['choices'][0]['message']['content']
        print(ret)

        targets = parse(ret)
        # get point clouds
        object_pcds_list = [self.env.get_points_by_name(object, camera_index=self.top_down_cam)]
        object_pcds_flat = [item for sublist in object_pcds_list for item in sublist]
        if material == 'rope':
            object_model_pcds = truncate_points(object_pcds_flat, fps_radius=0.02)
        elif material == 'cube':
            object_model_pcds = object_pcds_flat
        elif material == 'T_shape':
            object_model_pcds = object_pcds_flat
        elif material == 'granular':
            object_model_pcds = truncate_points(object_pcds_flat, fps_radius=0.02)
        object_points = np.concatenate(object_model_pcds, axis=0)
        # get new target specifications
        target_keys, target_values = list(targets.keys()), list(targets.values())
        target_indices = []
        destinations = []
        # rematch the target indices
        for target_index_orig in target_keys:
            image_point = self._ground_point(annotated_points[target_index_orig], W, H)
            annotated_3d_point = self.env.ground_position(self.top_down_cam, depth, image_point[0], image_point[1])
            # find nearest point in object_state
            dist = np.linalg.norm(object_points - annotated_3d_point, axis=1)
            target_index = np.argmin(dist)
            target_indices.append(target_index)
        # get the destinations
        for reference_index, array in target_values:
            if reference_index == -1:
                image_point = self._ground_point([W // 2, H // 2], W, H)
            else:
                image_point = self._ground_point(annotated_points[reference_index], W, H)
            annotated_3d_point = self.env.ground_position(self.top_down_cam, depth, image_point[0], image_point[1])
            # coordinate transformation
            delta = np.array([-array[1], array[0], array[2]], dtype=np.float32)
            delta /= 100
            destination = annotated_3d_point + delta
            destinations.append(destination)
        target_specification = (target_indices, destinations)

        # viz
        os.makedirs(f'{self.log_dir}/low_level', exist_ok=True)
        orig_save_image = Image.fromarray(cropped_obs[..., ::-1])
        orig_save_image.save(f'{self.log_dir}/low_level/orig_{call_depth}.png')
        annotated_save_image = Image.fromarray(annotated_image[..., ::-1])
        annotated_save_image.save(f'{self.log_dir}/low_level/annotated_{call_depth}.png')
        vis_image = self._view_specification(annotated_image, object_points, target_specification)
        save_image = Image.fromarray(vis_image[..., ::-1])
        save_image.save(f'{self.log_dir}/low_level/targets_{call_depth}.png')
        # save gpt response
        with open(f'{self.log_dir}/low_level/gpt_response_{call_depth}.txt', 'w') as f:
            f.write(ret)

        closed_loop_plan(
            object_points, target_specification, object, material, self.env,
            self.top_down_cam, self.tracking_cam, self.log_dir,
            track_as_state=False, target_pcd=self.target_pcd
        )
        if self.close_loop:
            self.__call__(object, material, instruction, call_depth=call_depth+1)

    def demo_call(self, img, instruction, material='cube'):
        predictor = GroundingSegmentPredictor(show_bbox=False, show_mask=True)
        H, W = img.shape[:2]

        # get the keypoints
        masks = predictor.mask_generation(img)
        masks.pop()

        annotated_points = []
        # keypoint hyperparameters
        num_per_obj = 8
        # Check before experiment
        radius = 200
        # for cubes, use the center to get the annotated points
        if material == 'cube':
            for mask in masks:
                positions = np.argwhere(mask)
                positions = positions[:, [1, 0]]
                keypoint = positions.mean(axis=0)
                annotated_points.append(keypoint)
        else:
            # use fps to get the annotated points
            for mask in masks:
                positions = np.argwhere(mask)
                positions = positions[:, [1, 0]]
                fps_1 = farthest_point_sampling(positions, num_per_obj)
                fps_2, _ = fps_rad_idx(fps_1, radius)
                annotated_points.extend(fps_2)
                # add center for each mask
                center = positions.mean(axis=0)
                annotated_points.append(center)
        # global radius
        global_radius = 150
        annotated_points, _ = fps_rad_idx(np.array(annotated_points), global_radius)

        # get the annotated image
        os.makedirs('test', exist_ok=True)
        annotated_image = get_annotated_image(img, annotated_points, debug=True, mask=masks)

        # GPT planning
        messages = self._build_prompt(annotated_image, instruction)
        ret = openai.ChatCompletion.create(
            messages=messages,
            temperature=self.config['temperature'],
            model=self.config['model'],
            max_tokens=self.config['max_tokens']
        )['choices'][0]['message']['content']
        print(ret)


def get_annotated_image(image, points, debug=False, mask=None):
    image = deepcopy(image)
    if debug:
        save_image = Image.fromarray(image[..., ::-1])
        save_image.save('test/0.png')

    # draw points
    for i, point in enumerate(points):
        x, y = point
        cv2.circle(image, (int(x), int(y)), 4, (0, 0, 255), -1)
        # cv2.circle(image, (int(x), int(y)), 4, (240, 32, 160), -1) # purple
        cv2.putText(image, f'P{i+1}', (int(x) + 3, int(y) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # add a center point for background reference
    H, W = image.shape[:2]
    x_center = W // 2
    y_center = H // 2
    cv2.circle(image, (x_center, y_center), 4, (0, 255, 0), -1)
    cv2.putText(image, 'C', (x_center + 3, y_center - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    if debug:
        save_image = Image.fromarray(image[..., ::-1])
        save_image.show()
        save_image.save('test/1.png')

    return image

def parse(response):
    """
    parse the python part of the response
    reference index could be -1, denoting the center point

    Returns:
      targets: {index: (index, array)}
    """
    targets = {}
    lines = response.split('\n')
    for line in lines:
        if '=' in line:
            line = line.strip()
            number_pattern = r'\d+'
            numbers = re.findall(number_pattern, line)
            letter_pattern = r'[a-zA-Z]'
            letters = re.findall(letter_pattern, line)
            target_index = eval(numbers[0]) - 1
            if len(letters) > 1 and letters[1] == 'p':
                reference_index = eval(numbers[1]) - 1
            else:
                reference_index = -1
            # find 3d array
            array_pattern = r'\[.*?\]'
            array = eval(re.findall(array_pattern, line)[0])
            targets[target_index] = (reference_index, array)

    return targets
