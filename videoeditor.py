import cv2

class Segmenter:
    def __init__(self, model: any) -> None:
        self.model = model

    def __call__(self, frame: any, size: tuple) -> any:
        import numpy as np
        frame = cv2.resize(frame, (256, 256))
        frame = np.expand_dims(frame, axis=0)
        frame = frame / 255.0
            
        mask = self.model.predict(frame, verbose=None)[0]

        mask = mask.astype(np.float32)
        mask = cv2.resize(mask, size)
        mask = np.expand_dims(mask, axis=-1)
        mask = np.clip(mask * 255, 0, 255).astype(np.uint8)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        return mask

class VideoEditor:
    PATH_ERROR_MESSAGE = 'The path to the video doesnt exist.'

    def _proccess_video(self, path: str, segmenter: Segmenter) -> None:
        video = cv2.VideoCapture(path)
        (width, height) = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                           int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = int(video.get(cv2.CAP_PROP_FPS))
        
        if self.out is not None: path = self.out + path[path.rfind('/'):]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f'{path[:path.rfind(".")]}_segmented.mp4', fourcc, 
                              fps, (width, height), True)
        
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        from tqdm import tqdm
        for _ in tqdm(range(length), desc=f'Please wait! Processing video: {path[(path.rfind("/") + 1):]}'):
            _, frame = video.read()
            frame = segmenter(frame, (width, height)) 
            out.write(frame)

        video.release()
        out.release()

    def __init__(self, path: str, out: str=None) -> None:
        import os
        if not os.path.exists(path): raise ValueError(self.PATH_ERROR_MESSAGE)
        if out is not None and not os.path.exists(out): os.makedirs(out)
        self.out = out

        import glob
        self.paths = []
        if os.path.isfile(path): self.paths.append(path)
        if os.path.isdir(path): self.paths = glob.glob(os.path.join(path, '*.*'))
        
    def __call__(self, model: any) -> None:
        segmenter = Segmenter(model)
        for path in self.paths:
            self._proccess_video(path, segmenter)