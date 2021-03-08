"""
Custom loss functions.

"""
import torch
DELTA_CLIP = 50.0

def original_loss(true, predict):
    """
        Loss function as implemented previously for solving the FBSDE
        Args:
            true: true target terminal cost
            predict: predicted terminal cost
        Output: loss
        """
    delta = predict - true
    loss = torch.where(delta.abs() < DELTA_CLIP, delta**2, 2*DELTA_CLIP*delta.abs()-DELTA_CLIP**2).mean() # use linear approximation outside the clipped range
    return loss


def V_Vx_custom_loss(V_true, V_predict, Vx_true, Vx_predict, Vxx_col_true, Vxx_col_predict,\
    weight_V, weight_Vx, weight_V_true, weight_Vx_true, weight_Vxx_col, weight_Vxx_col_true, \
    use_abs_loss):
    """
    Convex combination of predicted terminal cost and difference between predicted and true terminal cost
    Args:
        true: true target terminal cost
        predict: predicted terminal cost
        weight: weight for the different between prediction and target
    Output: convex combination of two losses
    """
    if use_abs_loss:
        V_dif_loss = (torch.abs(V_true - V_predict)).mean()
        V_abs_loss = (torch.abs(V_true)).mean()
        Vx_dif_loss = (torch.abs(Vx_true - Vx_predict)).mean()
        Vx_abs_loss = (torch.abs(Vx_true)).mean()
        Vxx_col_diff_loss = (torch.abs(Vxx_col_true - Vxx_col_predict)).mean()
    else:
        # print("summing true Vx and Vxx losses")
        V_dif_loss = original_loss(V_true, V_predict)
        V_abs_loss = (V_true**2).mean()
        
        Vx_dif_loss = original_loss(Vx_true, Vx_predict)
        Vx_abs_loss = (Vx_true**2).mean()
        # Vx_abs_loss = ((Vx_true**2).sum(dim=1)).mean()
        
        Vxx_col_diff_loss = original_loss(Vxx_col_true, Vxx_col_predict)
        Vxx_col_abs_loss = (Vxx_col_true**2).mean()
        # Vxx_col_abs_loss = ((Vxx_col_true**2).sum(dim=1)).mean()


    loss = weight_V*V_dif_loss + weight_V_true*V_abs_loss \
         + weight_Vx*Vx_dif_loss + weight_Vx_true*Vx_abs_loss \
         + weight_Vxx_col*Vxx_col_diff_loss \
         + weight_Vxx_col_true * Vxx_col_abs_loss

    return loss, V_dif_loss, V_abs_loss, Vx_dif_loss, Vx_abs_loss, Vxx_col_diff_loss, Vxx_col_abs_loss
