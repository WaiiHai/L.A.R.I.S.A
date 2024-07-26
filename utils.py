def init_run_option():
    from optparse import OptionParser
    command = OptionParser()
    command.add_option('-p', '--path', dest='path',
                      help='The path to the folder with video files')
    command.add_option('-m', '--model', dest='model',
                      help='The name of the model')
    command.add_option('-o', '--out', dest='output',
                      help='Folder where the processed video files will be placed')
    (options, args) = command.parse_args()

    return (options.model, options.path, options.output)

def display(data: any, size: int=1) -> None:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
        
    for x, images in enumerate(data):
        for i, image in enumerate(images[:size][:8]):
            plt.subplot(len(data), size, i + x * size + 1)
            plt.imshow(image, cmap='gray')
            plt.axis("off")
        
    import os
    image_path = os.path.join('video', 'image')
    plt.savefig(image_path)
    plt.close()