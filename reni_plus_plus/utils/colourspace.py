# import torch
# from typing import Optional

# def linear_to_sRGB(color, use_quantile=False, q: Optional[torch.Tensor] = None, clamp=True):
#     """Convert linear RGB to sRGB.

#     Args:
#         color: [..., 3]
#         use_quantile: Whether to use the 98th quantile to normalise the color values.

#         Returns:
#             color: [..., 3]
#     """
#     if use_quantile or q is not None:
#         if q is None:
#             q = torch.quantile(color.flatten(), 0.98)
#         color = color / q.expand_as(color)

#     color = torch.where(
#         color <= 0.0031308,
#         12.92 * color,
#         1.055 * torch.pow(torch.abs(color), 1 / 2.4) - 0.055,
#     )
#     if clamp:
#         color = torch.clamp(color, 0.0, 1.0)
#     return color

import torch
from typing import Optional

def linear_to_sRGB(color, use_quantile=False, q: Optional[torch.Tensor] = None, clamp=True):
    """Convert linear RGB to sRGB.

    Args:
        color: [..., 3]
        use_quantile: Whether to use the 98th quantile to normalise the color values.

        Returns:
            color: [..., 3]
    """
    # print("Min of color:", color.min())
    # print("Max of color:", color.max())
    # print("Any negative values:", (color < 0).any())
    # print("Any NaNs:", torch.isnan(color).any())
    # print("Any Infs:", torch.isinf(color).any())


    if use_quantile or q is not None:
        if q is None:
            q = torch.quantile(color.flatten(), 0.98)
        # Adding a small epsilon to avoid division by zero or very small numbers
        color = color / (q.expand_as(color) + 1e-8)

    color = torch.where(
        color <= 0.0031308,
        12.92 * color,
        1.055 * torch.pow(torch.abs(color) + 1e-8, 1 / 2.4) - 0.055,  # small constant added here
    )

    if torch.isnan(color).any():
        print("NaNs detected in color!")
        # Optionally handle NaNs
        color = torch.nan_to_num(color)  # Replace NaNs with zeros or another number

    if clamp:
        color = torch.clamp(color, 0.0, 1.0)
    return color

# import torch
# from typing import Optional

# def linear_to_sRGB(linear):
#     """
#     Converts linear RGB values to sRGB values using PyTorch.

#     Args:
#     linear (torch.Tensor): A tensor of linear RGB values. Values are expected to be in the range [0, 1].

#     Returns:
#     torch.Tensor: A tensor of sRGB values.
#     """
#     # Constants for the transformation
#     threshold = 0.0031308
#     scale_linear = 12.92
#     scale_gamma = 1.055
#     exponent_gamma = 1/2.4
#     offset_gamma = 0.055

#     # Apply the sRGB transformation with no in-place modifications
#     sRGB = torch.where(linear <= threshold,
#                        linear * scale_linear,
#                        scale_gamma * torch.pow(linear, exponent_gamma) - offset_gamma)

#     # Debugging: Check for NaNs or Infs
#     if torch.any(torch.isnan(sRGB)) or torch.any(torch.isinf(sRGB)):
#         print("NaNs or Infs detected in the sRGB values")

#     # Clamp the output to prevent values outside [0, 1]
#     sRGB = torch.clamp(sRGB, 0, 1)

#     return sRGB


# def normalize_color(color, eps=1e-8):
#     max_color = color.max()
#     return (color + eps) / (max_color + eps)

# def safe_pow(color, exponent, eps=1e-8):
#     return torch.pow(color + eps, exponent)

# def linear_to_sRGB(color, use_quantile=False, q: Optional[torch.Tensor] = None, clamp=True):
#     """Convert linear RGB to sRGB.

#     Args:
#         color: Tensor with shape [..., 3] representing RGB values.
#         use_quantile: Whether to use the 98th quantile to normalize the color values.
#         q: Optional tensor, specifies the quantile value for normalization if `use_quantile` is False.
#         clamp: Whether to clamp the output values to the range [0.0, 1.0].

#     Returns:
#         Tensor with shape [..., 3] representing sRGB values.
#     """
#     if use_quantile or q is not None:
#         if q is None:
#             q = torch.quantile(color.flatten(), 0.98) + 1e-8
#         color = color / q.expand_as(color)

#     # Normalize color to avoid extreme values
#     color = normalize_color(color)

#     color = torch.where(
#         color <= 0.0031308,
#         12.92 * color,
#         1.055 * safe_pow(color, 1 / 2.4) - 0.055
#     )
#     if clamp:
#         color = torch.clamp(color, 0.0, 1.0)
#     return color
