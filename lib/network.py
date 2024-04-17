from torch import nn
import torchvision


class CMUGuesserNet(nn.Module):
    def __init__(self, grid_size):
        '''Initializer for the MobileNet model to predict on the CMU campus images dataset

        1. load the mobilenet_v2 model (you will have to load weights depending on input pretrained)
        2. modify the final linear layer of the classifier to output 1 unit
        3. create a new final activation layer to turn the single output of the modified Mobilenet into a probability
            - we do this because MobileNet itself doesn't have any final activation layer
            - you will have to choose the appropriate activation for this last part

        Note: when loading the mobilenet model, set progress=False to avoid messing up the autograder

        Arguments:
            - grid_size: the size of the grid that the model will predict on (e.g. 10x10 grid)
        '''
        super().__init__()
        # NOTE: store your adjusted, pre-trained mobilenet model object to the self.mobilenet attribute
        #       and your final activation layer to self.finalActivation
        #       as it will be used for autograding (note: you will have to use these in your forward function)
        mobilenet = torchvision.models.mobilenet_v2(
            weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1)
        mobilenet.classifier[1] = nn.Sequential(
            nn.Linear(in_features=1280, out_features=grid_size *
                      grid_size, bias=True),
        )
        # mobilenet.classifier[1]= nn.Linear(in_features=1280, out_features=2, bias=True)

        self.mobilenet = mobilenet
        self.finalActivation = nn.Softmax(dim=1)

    def forward(self, x):
        '''Computes the model's predictions given data x

        Note: because this model is a convolutional neural network and uses convolutional layers to process
        the input images into features, we can directly pass in images into the model instead of having to
        flatten the entire image into a single row vector like homework 3 which used an multi-layer perceptron (i.e. fully connected neural network)

        Don't forget to pass the mobilenet output through the appropriate activation!

        Arguments:
            - x: a (batch_size, 3, SIZE, SIZE) floating point tensor (note: SIZE defined at top of notebook)
        Returns:
            - yhat: a (batch_size, 1) floating point tensor containing predictions on whether the each image is not hotdog or hotdog
        '''
        x = self.mobilenet.forward(x)
        return self.finalActivation(x)
