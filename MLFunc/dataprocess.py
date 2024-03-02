from torchvision import transforms

def img_to_tensor(img):
    """compose PIL image to tensor

    Args:
        img (PIL image)

    Returns:
        tensor (torch.tensor)
    """
    tf = transforms.Compose([transforms.ToTensor()])
    tensor = tf(img)
    return tensor