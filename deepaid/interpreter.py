class Interpreter(object):
    
    def __init__(self, model):
        r"""
        Initializes internal attack state.
        Arguments:
            model (torch.nn.Module): model to interpret.
            
        """
        self._training_mode = False
        self.early_stop = True
        self.verbose = False
        self.model = model
        # self.thres = thres
        self.model_name = str(model).split("(")[0]
        try:
            self.device = next(model.parameters()).device
        except:
            print('User Warning: Underlying model is not implemented with Pytorch')
            self.device = None

    def forward(self, *input):
        r"""
        It defines the computation performed at every call.
        Should be overridden by all subclasses.
        """
        raise NotImplementedError
        
    def __call__(self, *input, **kwargs):
        try:
            training_mode = self.model.training
            if self._training_mode:
                self.model.train()
            else:
                self.model.eval()
        except:
            print('User Warning: Underlying model is not implemented with Pytorch')

        interpretation = self.forward(*input, **kwargs)

        try:
            if training_mode:
                self.model.train()
        except:
            print('User Warning: Underlying model is not implemented with Pytorch')
#         return reference
        return interpretation