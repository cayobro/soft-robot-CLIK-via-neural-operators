import torch

class DeepONet(torch.nn.Module):
    """Deep Operator Network."""
    def __init__(self, params, dim_out=3):
        self._params = params
        layers_branch = self._params["layers_branch"]
        layers_trunk = self._params["layers_trunk"]
        if layers_branch[-1] != layers_trunk[-1]:
            raise ValueError("The dimension of the last layer of the branch and trunk network must be equal.")
        if layers_branch[-1] % dim_out != 0:
            raise ValueError("The dimension of the last layer of the branch and trunk network must divisable by dim_out.")
        super(DeepONet, self).__init__()
        self.layers_branch = torch.nn.ModuleList()
        for i in range(len(layers_branch) - 1):
            self.layers_branch.append(torch.nn.Linear(layers_branch[i], layers_branch[i + 1]))
            # Initialize weights using Xavier initialization
            torch.nn.init.xavier_normal_(self.layers_branch[-1].weight)
            torch.nn.init.zeros_(self.layers_branch[-1].bias)
        self.layers_trunk = torch.nn.ModuleList()
        for i in range(len(layers_trunk) - 1):
            self.layers_trunk.append(torch.nn.Linear(layers_trunk[i], layers_trunk[i + 1]))
            # Initialize weights using Xavier initialization
            torch.nn.init.xavier_normal_(self.layers_trunk[-1].weight)
            torch.nn.init.zeros_(self.layers_trunk[-1].bias)
        self.dim_out = dim_out
        self.p = int(layers_trunk[-1] / dim_out)

    @property
    def params(self):
        """Getter for param."""
        return self._params

    @params.setter
    def params(self, value):
        """Setter for param."""
        self._params = value

    def forward_branch(self, x):
        # Apply activation function to all layers except the last
        if self._params['activation'] == "tanh":
            for layer in self.layers_branch[:-1]:
                x = torch.tanh(layer(x))
        elif self._params['activation'] == "relu":
            for layer in self.layers_branch[:-1]:
                x = torch.relu(layer(x))
        else:
            raise ValueError("Activation function not supported.")
        return self.layers_trunk[-1](x)
    
    def forward_trunk(self, x):
        # Apply activation function to all layers except the last
        if self._params['activation'] == "tanh":
            for layer in self.layers_trunk[:-1]:
                x = torch.tanh(layer(x))
        elif self._params['activation'] == "relu":
            for layer in self.layers_trunk[:-1]:
                x = torch.relu(layer(x))
        else:
            raise ValueError("Activation function not supported.")
        return self.layers_trunk[-1](x)
    
    def forward(self, gamma, z):
        # Combine the outputs of the branch and trunk network
        # The outputs of the branch and trunk network have the dimension n_s x (p*dim_out), 
        # where n_s is the number of samples and dim_out is typically three
        # Bring the outputs into the form n_s x dim_out x p
        b = self.forward_branch(gamma).view(-1,self.dim_out,self.p)
        t = self.forward_trunk(z).view(-1,self.dim_out,self.p)
        # For both the branch and trunk network, we have for each data sample three vectors of dimension p
        # We compute the scalar product of the three vectors to arrive for each data sample at a three dimensional vector 
        return (b * t).sum(dim=-1)

        ## This is more complicated than imagines because all the data is scaled.
        # result = (b * t).sum(dim=-1)
        # enforce 0 at z=0
        # return torch.where(z == 0, torch.zeros_like(result), result)





	
        
    
        
        
        