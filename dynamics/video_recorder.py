import os
import multiprocessing as mp
from threadpoolctl import threadpool_limits
import cv2


class VideoRecorder(mp.Process):

    def __init__(self, index, camera, capture_fps, record_fps, record_time, save_path='recording/'):
        super().__init__()
        self.index = index
        self.capture_fps = capture_fps
        self.record_fps = record_fps
        self.record_time = record_time
        self.save_path = save_path
        self.camera = camera

    def run(self):
        # limit threads
        # threadpool_limits(1)
        # cv2.setNumThreads(1)

        i = self.index
        capture_fps = self.capture_fps
        record_fps = self.record_fps
        record_time = self.record_time
        save_path = self.save_path

        camera = self.camera

        out = None
        target_timestamp = None
        next_step_idx = 0

        os.makedirs(save_path, exist_ok=True)
        if os.path.exists(os.path.join(save_path, f'timestamps_{i}.txt')):
            os.remove(os.path.join(save_path, f'timestamps_{i}.txt'))
        f = open(os.path.join(self.save_path, f'timestamps_{i}.txt'), 'a')

        k = 2
        while self.alive:
            out = camera.get(out=out, k=k, index=i)

            if target_timestamp is None:
                target_timestamp = out['timestamp'][-1]

            for j in range(k-1, -1, -1):
                if abs(out['timestamp'][j] - target_timestamp) <= 0.5 * (1. / record_fps) or \
                        (j == 0 and out['timestamp'][j] > target_timestamp + 0.5 * (1. / record_fps)):
                    timestamps_str = " ".join([f"{t:.3f}" for t in out["timestamp"]])
                    print(f'Camera {i}, step {out["step_idx"][j]}, time {out["timestamp"][j]:.3f}', end='\t')
                    print(f'all steps {out["step_idx"]}, all timestamps {timestamps_str}')

                    cv2.imwrite(f'{save_path}/{next_step_idx:06}.jpg', out['color'][j])
                    cv2.imwrite(f'{save_path}/{next_step_idx:06}_depth.png', out['depth'][j])
                    # f.write(f'{timestamps_str}\n')
                    f.write(f'{out["timestamp"][j]:.3f}\n')
                    f.flush()

                    target_timestamp += 1. / record_fps
                    next_step_idx += 1

                    break

            if next_step_idx >= record_time * record_fps:
                f.close()
                self.alive = False

    def start(self):
        self.alive = True
        super().start()

    def stop(self):
        self.alive = False

    def join(self):
        super().join()
