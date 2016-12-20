function h=grad_sigmoid(z)
  h=sigmoid(z).*(1-sigmoid(z));
