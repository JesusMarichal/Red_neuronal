use burn::{
    module::Module,
    nn::{Dropout, DropoutConfig, Linear, LinearConfig, Relu},
    tensor::{backend::Backend, Tensor},
};
use crate::features::NUM_FEATURES;

/// Red densa 25 -> 32 -> 16 -> 2 con dropout.
///
/// El dropout de Burn se desactiva solo cuando el backend no tiene autodiff
/// (es decir, en `model.valid()`), así que la inferencia es determinista.
#[derive(Module, Debug)]
pub struct BaseballPredictor<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    out: Linear<B>,
    activation: Relu,
    dropout: Dropout,
}

impl<B: Backend> BaseballPredictor<B> {
    pub fn new(device: &B::Device, hidden1: usize, hidden2: usize, dropout: f64) -> Self {
        Self {
            fc1: LinearConfig::new(NUM_FEATURES, hidden1).init(device),
            fc2: LinearConfig::new(hidden1, hidden2).init(device),
            out: LinearConfig::new(hidden2, 2).init(device), // [gana visitante, gana local]
            activation: Relu::new(),
            dropout: DropoutConfig::new(dropout).init(),
        }
    }

    /// Devuelve logits sin normalizar: [batch, 2].
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.activation.forward(self.fc1.forward(input));
        let x = self.dropout.forward(x);
        let x = self.activation.forward(self.fc2.forward(x));
        let x = self.dropout.forward(x);
        self.out.forward(x)
    }
}
