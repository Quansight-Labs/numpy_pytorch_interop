import torch

def mandelbrot_pytorch(x_values, y_values, max_iter=200, device='cpu'):
    rx, iy = torch.meshgrid(x_values, y_values, indexing='xy') 

    x = rx.clone()
    y = iy.clone()
    mask = torch.zeros_like(x, device=device)
    for i in range(max_iter):
        x_prev = x
        y_prev = y
        x = x_prev**2 - y_prev**2 + rx
        y = 2*x_prev*y_prev + iy
        inside = torch.sqrt(x**2 + y**2) <= 2
        mask+=inside
    return mask

device = "cpu"

# Define the boundaries of the complex plane
xn = 450
yn = 375
xmin = -2.25
xmax = 0.75
ymin = -1.25
ymax = 1.25

# Create the grid of complex numbers
x_values = torch.linspace(xmin, xmax, xn, device=device, dtype=torch.float64)
y_values = torch.linspace(ymin, ymax, yn, device=device, dtype=torch.float64)

mandelbrot_compiled = torch.compile(mandelbrot_pytorch)

print(mandelbrot_compiled(x_values, y_values, 3))

