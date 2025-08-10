import numpy as np

class Categorical:
    """
    A categorical distribution parameterized by logits.
    Supports functionalities such as sampling, log probability computation, and entropy calculation.
    """
    def __init__(self, logits=None, validate_args=None):
        """
        Args:
            logits (np.ndarray): Unnormalized log probabilities.
            validate_args (bool): Whether to validate inputs (default: None).
        """
        if logits is None:
            raise ValueError("`logits` must be specified.")
        logits = np.asarray(logits, dtype=np.float32)
        if logits.ndim < 1:
            raise ValueError("`logits` parameter must be at least one-dimensional.")
        # Normalize logits to avoid overflow issues
        self.logits = logits - np.logaddexp.reduce(logits, axis=-1, keepdims=True)
        self.probs = np.exp(self.logits)  # Convert to probabilities
        self._num_events = self.logits.shape[-1]
        self._batch_shape = self.logits.shape[:-1]

    @property
    def batch_shape(self):
        return self._batch_shape

    @property
    def param_shape(self):
        return self.logits.shape

    @property
    def mean(self):
        return np.full(self._batch_shape, np.nan, dtype=np.float32)

    @property
    def mode(self):
        return np.argmax(self.probs, axis=-1)

    @property
    def variance(self):
        return np.full(self._batch_shape, np.nan, dtype=np.float32)

    def sample(self, sample_shape=()):
        """
        Draw samples from the distribution.

        Args:
            sample_shape (tuple): Shape of the samples to generate.

        Returns:
            np.ndarray: Sampled indices.
        """
        sample_shape = tuple(sample_shape)
        extended_shape = sample_shape + self._batch_shape
        flat_probs = self.probs.reshape(-1, self._num_events)
        samples = np.array([np.random.choice(self._num_events, size=sample_shape, p=p) for p in flat_probs])
        return samples.reshape(extended_shape)

    def log_prob(self, value):
        """
        Compute the log probability of given values.

        Args:
            value (np.ndarray): Values to compute log probabilities for.

        Returns:
            np.ndarray: Log probabilities of the input values.
        """
        value = np.asarray(value, dtype=np.int64)
        if value.shape != self._batch_shape:
            raise ValueError("`value` shape must match batch shape.")
        flat_logits = self.logits.reshape(-1, self._num_events)
        flat_values = value.flatten()
        flat_log_probs = flat_logits[np.arange(flat_logits.shape[0]), flat_values]
        return flat_log_probs.reshape(self._batch_shape)

    def entropy(self):
        """
        Compute the entropy of the distribution.

        Returns:
            np.ndarray: Entropy of the distribution.
        """
        min_real = np.finfo(self.logits.dtype).min
        logits = np.clip(self.logits, min_real, None)  # Avoid numerical issues
        p_log_p = logits * self.probs
        return -np.sum(p_log_p, axis=-1)

    def enumerate_support(self, expand=True):
        """
        Enumerate all possible values for the distribution.

        Args:
            expand (bool): Whether to expand the batch dimensions.

        Returns:
            np.ndarray: Support values.
        """
        support = np.arange(self._num_events, dtype=np.int64)
        if expand:
            support = np.expand_dims(support, axis=tuple(range(1, len(self._batch_shape) + 1)))
            support = np.broadcast_to(support, (self._num_events,) + self._batch_shape)
        return support


# Example usage
if __name__ == "__main__":
    logits = np.array([[2.0, 1.0, 0.0]])
    dist = Categorical(logits=logits)

    print("Probs:", dist.probs)
    print("Sample:", dist.sample(sample_shape=(3,)))
    print("Log Prob:", dist.log_prob(np.array([0])))
    print("Entropy:", dist.entropy())
    print("Mode:", dist.mode)
    print("Enumerate Support:", dist.enumerate_support(expand=True))
