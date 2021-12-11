# Instructions of building your own Interpreter

**Here is the instruction for customizing your own Interpreter if your models is not supported by our current implementation (e.g., different DL framework, different base DL model).** I believe this is not difficult because the high-level idea of DeepAID is actually very straightforward. Meanwhile, although DeepAID is a white-box interpretation method, it only needs the gradient information of DL models, so you only need to know how to get the gradient of your model. Below we will introduce the high-level idea for interpreting anomaly in DeepAID, and take the base model implemented by pytorch as an example to introduce some key steps for the implementation of the interpreter.

Take a look at our code structure. The code of the interpreter is under `DeepAID/deepaid/`. The base class of the interpreter is implemented in `interpreter.py`. Inheriting this base class can customize the interpreter for your model. We have implemented some interpreters in the folder named `interpreters/`. Here is the code of your customized interpreter. 

```python
class Customized_Interpreter(Interpreter):
    
    def __init__(self, 
                 model, # this is the base model to be interpreted 
                 hyperparameter1, # here are some hyper parameters used in the interpretation
                 hyperparameter2,
                 ... 
                 ):        
        super(Customized_Interpreter,self).__init__(model)
        
        self.hyperparameter1 = hyperparameter1 # initialize your hyperparameters
        ...
        
    def forward(self, 
                anomaly,  # to be interpreted
                constraint_params1, 
                constraint_params2, ...):
        
        """
        Here is the preprocessing and initialization of the reference
        """
        
        reference = anomaly.clone().detach()
        if secure_start:
            reference += np.random.normal(loc=0, scale=0.001, size=reference.shape) # (Optional) I.R.N to satisfy Robustness constraint
        reference.requires_grad = True
        optimizer = optim.SGD([reference], lr=self.lr) # set the optimizer for Interpreter 
        
        """
        Here is the initialization of Loss
        """
        
        for step in range(self.steps):
            """
            Here you need to define the Loss, which basically consists of Fidelity and Stability, see the following GIF or paper for these constraints.
            Loss = Fidelity_Loss + self.hyperparameterX * Stability_Loss
            """
        
            optimizer.zero_grad() 
            Loss.backward()
            optimizer.step()
            
        """
        Here are some clipping operations or selections to meet the requirements of Conciseness constraint
        """
        
        return {'index_inc': reference_index , 'value_inc': reference_value } # Here is the interpretation result, i.e., difference from reference
        
    def auto_calibration(self,): # (Optional) auto calibration of hyperparamters
        self.hyperparamters = ... # See Appendix for details
        
    def show_results(self, ): # (Optional) Show the interpretation result
        ...
    
    def other_optional_func(self, ):
        ...
        
```


![framework](framework.gif)
