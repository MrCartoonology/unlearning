import torch
from unlearn.trainers import apply_orthogonal_projection


def test_apply_orthogonal_projection_with_multiple_params():
    torch.manual_seed(42)

    # Create retain gradient (direction to be removed)
    retain_vector = torch.randn(6)
    retain_vector = retain_vector / retain_vector.norm()

    # Create model gradients as two parameter tensors
    g1 = retain_vector[:3].clone().detach()
    g2 = retain_vector[3:].clone().detach()
    model_gradients = [g1.clone(), g2.clone()]

    # Apply orthogonal projection
    apply_orthogonal_projection(model_gradients, [retain_vector.clone()])

    # Recombine projected gradients
    projected = torch.cat([p.view(-1) for p in model_gradients])
    
    # Check dot product with retain_vector is near zero
    dot_product = torch.dot(projected, retain_vector)
    assert torch.allclose(dot_product, torch.tensor(0.0), atol=1e-6), f"Dot product was {dot_product}"