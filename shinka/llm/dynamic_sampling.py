import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Union, Sequence, List, Any
from scipy.special import logsumexp
from rich.table import Table
from rich.console import Console
import rich.box

Arm = Union[int, str]
Subset = Optional[Union[np.ndarray, Sequence[Arm]]]


def _logadd(x_log, y_log, w1=1.0, w2=1.0):
    x = np.asarray(x_log, dtype=float) + np.log(w1)
    y = np.asarray(y_log, dtype=float) + np.log(w2)
    a = np.stack([x, y], axis=0)
    return logsumexp(a, axis=0)


def _logdiffexp(a_log, b_log):
    a = np.asarray(a_log, float)
    b = np.asarray(b_log, float)
    d = a - b
    with np.errstate(over="ignore", invalid="ignore"):
        v = a + np.log1p(-np.exp(-d))
    return np.where(d >= 0, v, -np.inf)


def _logexpm1(z):
    z = np.asarray(z, dtype=float)
    return np.where(z > 50.0, z, np.log(np.expm1(z)))


class BanditBase(ABC):
    def __init__(
        self,
        n_arms: Optional[int] = None,
        seed: Optional[int] = None,
        arm_names: Optional[List[str]] = None,
        auto_decay: Optional[float] = None,
        shift_by_baseline: bool = True,
        shift_by_parent: bool = True,
    ):
        self.rng = np.random.default_rng(seed)

        if arm_names is None and n_arms is None:
            raise ValueError("provide n_arms or arm_names")
        if arm_names is not None:
            if n_arms is not None and int(n_arms) != len(arm_names):
                raise ValueError("len(arm_names) must equal n_arms")
            self._arm_names = list(arm_names)
            self._name_to_idx = {n: i for i, n in enumerate(self._arm_names)}
            self._n_arms = len(self._arm_names)
        else:
            self._arm_names = None
            self._name_to_idx = {}
            self._n_arms = int(n_arms)

        self._baseline = 0.0
        self._shift_by_baseline = bool(shift_by_baseline)
        self._shift_by_parent = bool(shift_by_parent)
        if auto_decay is not None and not (0.0 < auto_decay <= 1.0):
            raise ValueError("auto_decay must be in (0, 1]")
        self._auto_decay = auto_decay

    @property
    def n_arms(self) -> int:
        return self._n_arms

    def set_baseline_score(
        self,
        baseline: float,
    ) -> None:
        self._baseline = float(baseline)

    def _resolve_arm(self, arm: Arm) -> int:
        # allows updating by int index or string name
        if isinstance(arm, int):
            return int(arm)
        if self._arm_names is None:
            try:
                return int(arm)
            except Exception as e:
                raise ValueError("string arm requires arm_names") from e
        if arm not in self._name_to_idx:
            raise ValueError(f"unknown arm name '{arm}'")
        return self._name_to_idx[arm]

    def _resolve_subset(self, subset: Subset) -> np.ndarray:
        if subset is None:
            return np.arange(self.n_arms, dtype=np.int64)
        if isinstance(subset, np.ndarray) and np.issubdtype(subset.dtype, np.integer):
            return subset.astype(np.int64)
        idxs = [self._resolve_arm(a) for a in subset]
        return np.asarray(idxs, dtype=np.int64)

    def _maybe_decay(self) -> None:
        if self._auto_decay is not None:
            self.decay(self._auto_decay)

    @abstractmethod
    def update_submitted(
        self,
        arm: Arm,
    ) -> float:
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        arm: Arm,
        reward: Optional[float],
        baseline: Optional[float] = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def posterior(
        self,
        subset: Subset = None,
        samples: Optional[int] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def decay(self, factor: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def print_summary(self) -> None:
        raise NotImplementedError


class AsymmetricUCB(BanditBase):
    # asymmetric ucb1 with ε-exploration and adaptive scaling
    def __init__(
        self,
        n_arms: Optional[int] = None,
        seed: Optional[int] = None,
        exploration_coef: float = 1.0,
        epsilon: float = 0.2,
        arm_names: Optional[List[str]] = None,
        auto_decay: Optional[float] = 0.95,
        shift_by_baseline: bool = True,
        shift_by_parent: bool = True,
        adaptive_scale: bool = True,
        asymmetric_scaling: bool = True,
        exponential_base: Optional[float] = 1.0,
    ):
        super().__init__(
            n_arms=n_arms,
            seed=seed,
            arm_names=arm_names,
            auto_decay=auto_decay,
            shift_by_baseline=shift_by_baseline,
            shift_by_parent=shift_by_parent,
        )
        if asymmetric_scaling:
            assert shift_by_baseline or shift_by_parent, (
                "asymmetric scaling requires at least one of "
                "shift_by_baseline or shift_by_parent to be True"
            )
        if not (0.0 <= epsilon <= 1.0):
            raise ValueError("epsilon must be in [0, 1]")
        self.c = float(exploration_coef)
        self.epsilon = float(epsilon)
        self.adaptive_scale = bool(adaptive_scale)
        self.asymmetric_scaling = bool(asymmetric_scaling)
        self.exponential_base = exponential_base

        self.use_exponential_scaling = self.exponential_base is not None

        # if none, no exponential scaling
        if self.exponential_base is not None:
            assert self.exponential_base > 0.0, "exponential_base must be > 0"
            self.exponential_base = float(exponential_base)

        n = self.n_arms
        self.n_submitted = np.zeros(n, dtype=np.float64)
        self.n_completed = np.zeros(n, dtype=np.float64)
        if self.use_exponential_scaling:
            self.s = np.full(n, -np.inf, dtype=np.float64)
        else:
            self.s = np.zeros(n, dtype=np.float64)
        self.divs = np.zeros(n, dtype=np.float64)

        if self.asymmetric_scaling:
            if self.use_exponential_scaling:
                self._obs_max = -np.inf
                self._obs_min = -np.inf
            else:
                self._obs_min = 0.0
                self._obs_max = 0.0
        else:
            self._obs_max = -np.inf
            self._obs_min = np.inf

    @property
    def n(self) -> np.ndarray:
        return np.maximum(self.n_submitted, self.n_completed)

    def _add_to_reward(self, r: float, value: float, coeff_r=1, coeff_value=1) -> float:
        if self.use_exponential_scaling:
            out, sign = logsumexp(
                [r, value],
                b=[coeff_r, coeff_value],
                return_sign=True,
            )
        else:
            out = coeff_r * r + coeff_value * value
        return out

    def _multiply_reward(self, r: float, value: float) -> float:
        if self.use_exponential_scaling:
            assert value > 0, "Multipliers in log space must be > 0"
            out = r + np.log(value)
        else:
            out = r * value
        return out

    def _mean(self) -> np.ndarray:
        denom = np.maximum(self.divs, 1e-7)
        if self.use_exponential_scaling:
            return self.s - np.log(denom)
        else:
            return self.s / denom

    def _update_obs_range(self, r: float) -> None:
        if r > self._obs_max:
            self._obs_max = r
        if not (self.use_exponential_scaling and self.asymmetric_scaling):
            if r < self._obs_min:
                self._obs_min = r

    def _have_obs_range(self) -> bool:
        if self.use_exponential_scaling and self.asymmetric_scaling:
            return np.isfinite(self._obs_max)
        return (
            np.isfinite(self._obs_min)
            and np.isfinite(self._obs_max)
            and (self._obs_max - self._obs_min) > 0.0
        )

    def _impute_worst_reward(self) -> float:
        if self.asymmetric_scaling:
            return -np.inf if self.use_exponential_scaling else 0.0

        seen = self.n > 0
        if not np.any(seen):
            return 0.0

        denom = np.maximum(self.divs[seen], 1e-7)
        mu = self.s[seen] / denom
        mu_min = float(mu.min())
        if mu.size >= 2:
            s = float(mu.std(ddof=1))
            sigma = 1.0 if (not np.isfinite(s) or s <= 0.0) else s
        else:
            sigma = 1.0
        return mu_min - sigma

    def _normalized_means(self, idx):
        if not self.adaptive_scale or not self._have_obs_range():
            m = self._mean()[idx]
            return np.exp(m) if self.use_exponential_scaling else m
        elif self.use_exponential_scaling and self.asymmetric_scaling:
            mlog = self._mean()[idx]
            return np.exp(mlog - self._obs_max)
        elif self.use_exponential_scaling:
            means_log = self._mean()[idx]
            rng_log = _logdiffexp(self._obs_max, self._obs_min)
            num_log = _logdiffexp(means_log, self._obs_min)
            return np.exp(num_log - rng_log)
        else:
            means = self._mean()[idx]
            rng = max(self._obs_max - self._obs_min, 1e-9)
            return (means - self._obs_min) / rng

    def update_submitted(
        self,
        arm: Arm,
    ) -> float:
        arm = self._resolve_arm(arm)
        self.n_submitted[arm] += 1.0
        return self.n[arm]

    def update(self, arm, reward, baseline=None):
        i = self._resolve_arm(arm)
        is_real = reward is not None
        r_raw = float(reward) if is_real else self._impute_worst_reward()

        if self._shift_by_parent and self._shift_by_baseline:
            baseline = (
                self._baseline if baseline is None else max(baseline, self._baseline)
            )
        elif self._shift_by_baseline:
            baseline = self._baseline
        elif not self._shift_by_parent:
            baseline = 0.0
        if baseline is None:
            raise ValueError("baseline required when shifting is active")

        r = r_raw - baseline

        if self.asymmetric_scaling:
            r = max(r, 0.0)

        self.divs[i] += 1.0
        self.n_completed[i] += 1.0

        if self.use_exponential_scaling and self.asymmetric_scaling:
            z = r * self.exponential_base
            if self._shift_by_baseline:
                contrib_log = _logexpm1(z)
            else:
                contrib_log = z
            self.s[i] = _logadd(self.s[i], contrib_log)
            if self.adaptive_scale and is_real:
                self._update_obs_range(contrib_log)
        else:
            self.s[i] += r
            if self.adaptive_scale and is_real:
                self._update_obs_range(r)

        self._maybe_decay()
        return r, baseline

    def posterior(self, subset=None, samples=None):
        idx = self._resolve_subset(subset)
        if samples is None or int(samples) <= 1:
            idx = self._resolve_subset(subset)
            n_sub = self.n[idx]
            probs = np.zeros(self._n_arms, dtype=np.float64)

            if np.all(n_sub <= 0.0):
                p = np.ones(idx.size) / idx.size
                probs[idx] = p
                return probs

            unseen = np.where(n_sub <= 0.0)[0]
            if unseen.size > 0:
                p = np.ones(unseen.size) / unseen.size
                probs[idx[unseen]] = p
                return probs

            t = float(self.n.sum())
            base = self._normalized_means(idx)
            num = 2.0 * np.log(max(t, 2.0))
            bonus = self.c * np.sqrt(num / n_sub)
            scores = base + bonus

            winners = np.where(scores == scores.max())[0]
            rem = idx.size - winners.size
            p_sub = np.zeros(idx.size, dtype=np.float64)
            if rem == 0:
                p_sub[:] = 1.0 / idx.size
            else:
                p_sub[winners] = (1.0 - self.epsilon) / winners.size
                mask = np.ones(idx.size, dtype=bool)
                mask[winners] = False
                p_sub[mask] = self.epsilon / rem
            probs[idx] = p_sub
            return probs
        else:
            return self._posterior_batch(idx, samples)

    def _posterior_batch(self, idx: np.ndarray, k: int) -> np.ndarray:
        A = idx.size
        probs = np.zeros(self._n_arms, dtype=np.float64)
        if k <= 0 or A == 0:
            return probs

        n_sub = self.n[idx].astype(np.float64)
        v = np.zeros(A, dtype=np.int64)

        if np.all(n_sub <= 0.0):
            p = np.ones(A, dtype=np.float64) / A
            probs[idx] = p
            return probs

        unseen = np.where(n_sub <= 0.0)[0]
        if unseen.size > 0:
            if k >= unseen.size:
                v[unseen] += 1
                k -= unseen.size
            else:
                take = int(k)
                sel = self.rng.choice(unseen, size=take, replace=False)
                v[sel] += 1
                k = 0
            if k == 0:
                alloc = v.astype(np.float64)
                probs[idx] = alloc / alloc.sum()
                return probs

        base = self._normalized_means(idx)
        t0 = float(self.n.sum())
        step = int(v.sum()) + 1

        # simulate remaining k virtual pulls with epsilon-greedy
        while k > 0:
            num = 2.0 * np.log(max(t0 + step, 2.0))
            den = np.maximum(n_sub + v, 1.0)
            scores = base + self.c * np.sqrt(num / den)

            winners = np.where(scores == scores.max())[0]
            p = np.zeros(A, dtype=np.float64)
            if winners.size == A:
                p[:] = 1.0 / A
            else:
                p[winners] = (1.0 - self.epsilon) / winners.size
                mask = np.ones(A, dtype=bool)
                mask[winners] = False
                others = np.where(mask)[0]
                if others.size > 0:
                    p[others] = self.epsilon / others.size

            i = int(self.rng.choice(A, p=p))
            v[i] += 1
            step += 1
            k -= 1

        alloc = v.astype(np.float64)
        probs[idx] = alloc / alloc.sum()
        return probs

    def decay(self, factor: float) -> None:
        if not (0.0 < factor <= 1.0):
            raise ValueError("factor must be in (0, 1]")
        self.divs = self.divs * factor
        one_minus_factor = 1.0 - factor
        if self.use_exponential_scaling and self.asymmetric_scaling:
            # shrink in exp space to match original score scale
            s = self.s
            log1p_term = np.where(
                s > 0.0,
                s + np.log(one_minus_factor + np.exp(-s)),
                np.log1p(one_minus_factor * np.exp(s)),
            )
            self.s = s + np.log(factor) - log1p_term

            if self.adaptive_scale and np.isfinite(self._obs_max):
                means_log = self._mean()
                mmax = float(np.max(means_log))
                om = self._obs_max
                log1p_obs = (
                    om + np.log(one_minus_factor + np.exp(-om))
                    if om > 0.0
                    else np.log1p(one_minus_factor * np.exp(om))
                )
                obs_new = om + np.log(factor) - log1p_obs
                self._obs_max = max(obs_new, mmax)
        else:
            self.s = self.s * factor
            if self.adaptive_scale and self._have_obs_range():
                means = self._mean()
                self._obs_max = max(
                    self._obs_max * factor + one_minus_factor * np.max(means),
                    np.max(means),
                )
                self._obs_min = min(
                    self._obs_min * factor + one_minus_factor * np.min(means),
                    np.min(means),
                )

    def print_summary(self) -> None:
        names = self._arm_names or [str(i) for i in range(self._n_arms)]
        post = self.posterior()
        n = self.n.astype(int)
        mean = self._mean()
        if self.use_exponential_scaling:
            mean_disp = mean  # keep in log space
            mean_label = "log mean"
        else:
            mean_disp = mean
            mean_label = "mean"
        idx = np.arange(self._n_arms)

        # exploitation and exploration components
        exploitation = self._normalized_means(idx)
        t = float(self.n.sum())
        num = 2.0 * np.log(max(t, 2.0))
        n_sub = np.maximum(self.n[idx], 1.0)
        exploration = self.c * np.sqrt(num / n_sub)
        score = exploitation + exploration

        # Create header information
        exp_base_str = (
            f"{self.exponential_base:.3f}"
            if self.exponential_base is not None
            else "None"
        )
        header_info = (
            f"AsymmetricUCB (c={self.c:.3f}, eps={self.epsilon:.3f}, "
            f"adaptive={self.adaptive_scale}, asym={self.asymmetric_scaling}, "
            f"exp_base={exp_base_str}, shift_base={self._shift_by_baseline}, "
            f"shift_parent={self._shift_by_parent}, "
            f"log_sum={self.use_exponential_scaling})"
        )

        additional_info = []
        if self._auto_decay is not None:
            additional_info.append(f"auto_decay={self._auto_decay:.3f}")
        additional_info.append(f"baseline={self._baseline:.6f}")

        if np.isfinite(self._obs_min) and np.isfinite(self._obs_max):
            if self.use_exponential_scaling:
                obs_min = np.exp(self._obs_min)
                obs_max = np.exp(self._obs_max)
            else:
                obs_min = self._obs_min
                obs_max = self._obs_max
            rng = obs_max - obs_min
            additional_info.append(
                f"obs_range=[{obs_min:.6f},{obs_max:.6f}] (w={rng:.6f})"
            )

        # Create rich table
        table = Table(
            title=header_info,
            box=rich.box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
            width=120,  # Match display.py table width
        )

        # Add columns
        table.add_column("arm", style="white", width=24)
        table.add_column("n", justify="right", style="green")
        table.add_column("div", justify="right", style="yellow")
        table.add_column(mean_label, justify="right", style="blue")
        table.add_column("exploit", justify="right", style="magenta")
        table.add_column("explore", justify="right", style="cyan")
        table.add_column("score", justify="right", style="bold white")
        table.add_column("post", justify="right", style="bright_green")

        # Add rows
        for i, name in enumerate(names):
            # Split name by "/" and take last part, then last 25 chars
            if isinstance(name, str):
                display_name = name.split("/")[-1][-25:]
            else:
                display_name = str(name)
            table.add_row(
                display_name,
                f"{n[i]:d}",
                f"{self.divs[i]:.3f}",
                f"{mean_disp[i]:.4f}",
                f"{exploitation[i]:.4f}",
                f"{exploration[i]:.4f}",
                f"{score[i]:.4f}",
                f"{post[i]:.4f}",
            )

        # Print directly to console
        console = Console()
        console.print(table)


class ThompsonSamplingBandit(BanditBase):
    """
    Thompson Sampling bandit algorithm using Beta distributions.
    
    This algorithm maintains Beta(alpha_i, beta_i) distributions for each arm,
    which are updated based on rewards. It provides good exploration-exploitation
    balance and handles non-stationary rewards well through the decay mechanism.
    
    The algorithm works by:
    1. Sampling from Beta(alpha_i, beta_i) for each arm
    2. Selecting the arm with highest sampled value
    3. Updating alpha_i (success) or beta_i (failure) based on reward
    """
    
    def __init__(
        self,
        n_arms: Optional[int] = None,
        seed: Optional[int] = None,
        arm_names: Optional[List[str]] = None,
        auto_decay: Optional[float] = 0.99,  # More aggressive decay for non-stationary
        shift_by_baseline: bool = True,
        shift_by_parent: bool = True,
        prior_alpha: float = 1.0,  # Prior success count
        prior_beta: float = 1.0,   # Prior failure count
        reward_mapping: str = "adaptive",  # "adaptive", "direct", "sigmoid"
        reward_scaling: float = 1.0,  # Scale rewards for adaptive/sigmoid mapping
        **kwargs: Any,
    ):
        super().__init__(
            n_arms=n_arms,
            seed=seed,
            arm_names=arm_names,
            auto_decay=auto_decay,
            shift_by_baseline=shift_by_baseline,
            shift_by_parent=shift_by_parent,
        )
        
        # Validate parameters
        if prior_alpha <= 0 or prior_beta <= 0:
            raise ValueError("prior_alpha and prior_beta must be > 0")
        if reward_scaling <= 0:
            raise ValueError("reward_scaling must be > 0")
        if reward_mapping not in ["adaptive", "direct", "sigmoid"]:
            raise ValueError("reward_mapping must be 'adaptive', 'direct', or 'sigmoid'")
            
        self.prior_alpha = float(prior_alpha)
        self.prior_beta = float(prior_beta)
        self.reward_mapping = reward_mapping
        self.reward_scaling = float(reward_scaling)
        
        # For adaptive mapping
        self.reward_history = []
        self.baseline_history = []
        
        # Initialize Beta parameters for each arm
        n = self.n_arms
        self.alpha = np.full(n, self.prior_alpha, dtype=np.float64)
        self.beta = np.full(n, self.prior_beta, dtype=np.float64)
        self.n_submitted = np.zeros(n, dtype=np.float64)
        self.n_completed = np.zeros(n, dtype=np.float64)
        
    @property
    def n(self) -> np.ndarray:
        return np.maximum(self.n_submitted, self.n_completed)
    
    def _sigmoid(self, x: float) -> float:
        """Sigmoid function to map rewards to [0,1] range."""
        return 1.0 / (1.0 + np.exp(-x * self.reward_scaling))
    
    def _adaptive_mapping(self, reward_diff: float, baseline: float) -> float:
        """
        Adaptive reward mapping based on observed reward distribution.
        
        This approach automatically adapts to the reward range and avoids
        the information compression problem of sigmoid mapping.
        """
        # Store history for adaptation
        if len(self.reward_history) >= 1000:  # Keep last 1000
            self.reward_history = self.reward_history[-500:]
            self.baseline_history = self.baseline_history[-500:]
            
        self.reward_history.append(reward_diff + baseline)  # Store original reward
        self.baseline_history.append(baseline)
        
        # For first few samples, assume rewards in [0,1] with given baseline
        if len(self.reward_history) < 20:
            # Simple linear mapping assuming typical [0,1] reward range
            prob = (reward_diff + 0.5)  # Assume baseline ~0.5, map [-0.5,0.5] to [0,1]
            return np.clip(prob, 0.0, 1.0)
        
        # After enough samples, use percentile-based adaptive mapping
        recent_rewards = np.array(self.reward_history[-100:])  # Last 100 rewards
        
        # Use robust percentiles (5th and 95th) to avoid outliers
        r_min = np.percentile(recent_rewards, 5)
        r_max = np.percentile(recent_rewards, 95)
        
        # If all rewards are similar, use neutral probability
        if r_max - r_min < 1e-6:
            return 0.5
        
        # Linear normalization to [0,1] based on observed range
        original_reward = reward_diff + baseline
        normalized = (original_reward - r_min) / (r_max - r_min)
        
        # Apply reward_scaling as "sharpening" factor
        if self.reward_scaling != 1.0 and 0 < normalized < 1:
            # Sharpen distribution around 0.5
            if normalized > 0.5:
                normalized = 0.5 + (normalized - 0.5) * self.reward_scaling
            else:
                normalized = 0.5 - (0.5 - normalized) * self.reward_scaling
        
        return np.clip(normalized, 0.0, 1.0)
    
    def _direct_mapping(self, reward_diff: float, baseline: float) -> float:
        """
        Direct linear mapping assuming rewards are in [0,1] range.
        
        This is the simplest and most transparent mapping for cases where
        you know rewards are already in a reasonable range.
        """
        # Assume baseline is middle of reward range (e.g., 0.5 for [0,1])
        # Map reward_diff from [-baseline, 1-baseline] to [0,1]
        expected_min = -baseline
        expected_max = 1.0 - baseline
        
        if expected_max - expected_min < 1e-6:
            return 0.5
            
        prob = (reward_diff - expected_min) / (expected_max - expected_min)
        
        # Apply reward_scaling
        if self.reward_scaling != 1.0 and 0 < prob < 1:
            prob = prob ** (1.0 / self.reward_scaling)
            
        return np.clip(prob, 0.0, 1.0)
    
    def _reward_to_success_probability(self, reward_diff: float, baseline: float) -> float:
        """Convert reward difference to success probability using selected mapping."""
        
        if self.reward_mapping == "adaptive":
            return self._adaptive_mapping(reward_diff, baseline)
        elif self.reward_mapping == "direct": 
            return self._direct_mapping(reward_diff, baseline)
        elif self.reward_mapping == "sigmoid":
            return self._sigmoid(reward_diff)
        else:
            raise ValueError(f"Unknown reward_mapping: {self.reward_mapping}")
    
    def update_submitted(self, arm: Arm) -> float:
        arm_idx = self._resolve_arm(arm)
        self.n_submitted[arm_idx] += 1.0
        return self.n[arm_idx]
    
    def update(
        self, 
        arm: Arm, 
        reward: Optional[float], 
        baseline: Optional[float] = None
    ) -> tuple[float, float]:
        arm_idx = self._resolve_arm(arm)
        is_real = reward is not None
        
        # Handle baseline shifting (same logic as AsymmetricUCB)
        if self._shift_by_parent and self._shift_by_baseline:
            baseline = (
                self._baseline if baseline is None else max(baseline, self._baseline)
            )
        elif self._shift_by_baseline:
            baseline = self._baseline
        elif not self._shift_by_parent:
            baseline = 0.0
        if baseline is None:
            raise ValueError("baseline required when shifting is active")
            
        # Process reward
        if is_real:
            r_raw = float(reward)
        else:
            # For failed/missing rewards, use a pessimistic value
            r_raw = baseline - 1.0  # Penalty for failures
            
        r = r_raw - baseline
        
        # Convert reward to success probability using sigmoid
        # Positive rewards -> higher probability of success
        success_prob = self._sigmoid(r)
        
        # Update Beta parameters
        # We treat this as a Bernoulli trial with success probability based on reward
        if is_real or r < 0:  # Only update for real rewards or penalties
            self.alpha[arm_idx] += success_prob
            self.beta[arm_idx] += (1.0 - success_prob)
            
        self.n_completed[arm_idx] += 1.0
        
        # Apply decay
        self._maybe_decay()
        
        return r, baseline
    
    def posterior(
        self, 
        subset: Subset = None, 
        samples: Optional[int] = None, 
        **kwargs: Any
    ) -> np.ndarray:
        idx = self._resolve_subset(subset)
        probs = np.zeros(self._n_arms, dtype=np.float64)
        
        if samples is None or int(samples) <= 1:
            # Single sample from each Beta distribution, then select best
            beta_samples = np.zeros(len(idx))
            for i, arm_idx in enumerate(idx):
                beta_samples[i] = self.rng.beta(
                    self.alpha[arm_idx], 
                    self.beta[arm_idx]
                )
            
            # Select arm with highest sample (Thompson Sampling)
            best_idx = np.argmax(beta_samples)
            probs[idx[best_idx]] = 1.0
            
        else:
            # Multi-sample version: sample multiple times and compute selection frequencies
            samples = int(samples)
            selection_counts = np.zeros(len(idx), dtype=int)
            
            for _ in range(samples):
                beta_samples = np.zeros(len(idx))
                for i, arm_idx in enumerate(idx):
                    beta_samples[i] = self.rng.beta(
                        self.alpha[arm_idx], 
                        self.beta[arm_idx]
                    )
                best_idx = np.argmax(beta_samples)
                selection_counts[best_idx] += 1
            
            # Convert counts to probabilities
            selection_probs = selection_counts.astype(np.float64) / samples
            probs[idx] = selection_probs
            
        return probs
    
    def decay(self, factor: float) -> None:
        """
        Apply decay to Beta parameters to handle non-stationary rewards.
        
        This moves the parameters closer to the prior, effectively "forgetting"
        old observations to adapt to changing reward distributions.
        """
        if not (0.0 < factor <= 1.0):
            raise ValueError("factor must be in (0, 1]")
            
        # Exponential decay towards prior
        # alpha = factor * alpha + (1 - factor) * prior_alpha
        self.alpha = factor * self.alpha + (1.0 - factor) * self.prior_alpha
        self.beta = factor * self.beta + (1.0 - factor) * self.prior_beta
        
        # Also decay submission counts
        self.n_submitted *= factor
        self.n_completed *= factor
    
    def print_summary(self) -> None:
        """Print a summary table of the Thompson Sampling bandit state."""
        names = self._arm_names or [str(i) for i in range(self._n_arms)]
        post = self.posterior()
        n = self.n.astype(int)
        
        # Compute statistics
        mean = self.alpha / (self.alpha + self.beta)  # Beta distribution mean
        variance = (self.alpha * self.beta) / ((self.alpha + self.beta)**2 * (self.alpha + self.beta + 1))
        std = np.sqrt(variance)
        
        # Create header information
        header_info = (
            f"ThompsonSamplingBandit (prior_α={self.prior_alpha:.2f}, "
            f"prior_β={self.prior_beta:.2f}, reward_scale={self.reward_scaling:.2f}, "
            f"shift_base={self._shift_by_baseline}, shift_parent={self._shift_by_parent})"
        )
        
        additional_info = []
        if self._auto_decay is not None:
            additional_info.append(f"auto_decay={self._auto_decay:.3f}")
        additional_info.append(f"baseline={self._baseline:.6f}")
        
        # Create rich table
        table = Table(
            title=header_info,
            box=rich.box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
            width=120,
        )
        
        # Add columns
        table.add_column("arm", style="white", width=24)
        table.add_column("n", justify="right", style="green")
        table.add_column("α", justify="right", style="yellow") 
        table.add_column("β", justify="right", style="yellow")
        table.add_column("mean", justify="right", style="blue")
        table.add_column("std", justify="right", style="magenta")
        table.add_column("post", justify="right", style="bright_green")
        
        # Add rows
        for i, name in enumerate(names):
            # Split name by "/" and take last part, then last 24 chars
            if isinstance(name, str):
                display_name = name.split("/")[-1][-24:]
            else:
                display_name = str(name)
            table.add_row(
                display_name,
                f"{n[i]:d}",
                f"{self.alpha[i]:.3f}",
                f"{self.beta[i]:.3f}",
                f"{mean[i]:.4f}",
                f"{std[i]:.4f}",
                f"{post[i]:.4f}",
            )
        
        # Print additional info
        if additional_info:
            info_text = " | ".join(additional_info)
            table.caption = info_text
            
        # Print directly to console
        console = Console()
        console.print(table)


class FixedSampler(BanditBase):
    # samples from fixed prior probabilities; no learning or decay
    def __init__(
        self,
        n_arms: Optional[int] = None,
        seed: Optional[int] = None,
        prior_probs: Optional[np.ndarray] = None,
        arm_names: Optional[List[str]] = None,
        auto_decay: Optional[float] = None,
        shift_by_baseline: bool = True,
        shift_by_parent: bool = True,
        **kwargs: Any,
    ):
        super().__init__(
            n_arms=n_arms,
            seed=seed,
            arm_names=arm_names,
            auto_decay=auto_decay,
            shift_by_baseline=shift_by_baseline,
            shift_by_parent=shift_by_parent,
        )
        n = self.n_arms
        if prior_probs is None:
            self.p = np.full(n, 1.0 / n, dtype=np.float64)
        else:
            p = np.asarray(prior_probs, dtype=np.float64)
            if p.ndim != 1 or p.size != n:
                raise ValueError("prior_probs must be length n_arms")
            if np.any(p < 0.0):
                raise ValueError("prior_probs must be >= 0")
            s = p.sum()
            if s <= 0.0:
                raise ValueError("prior_probs must sum to > 0")
            self.p = p / s

    def update_submitted(
        self,
        arm: Arm,
    ) -> float:
        return 0.0

    def update(
        self,
        arm: Arm,
        reward: Optional[float],
        baseline: Optional[float] = None,
    ) -> tuple[float, float]:
        self._maybe_decay()
        return 0.0, baseline

    def posterior(
        self,
        subset: Subset = None,
        samples: Optional[int] = None,
    ) -> np.ndarray:
        # return fixed selection probabilities per arm
        if subset is None:
            return self.p.copy()
        idx = self._resolve_subset(subset)
        probs = self.p[idx]
        s = probs.sum()
        if s <= 0.0:
            raise ValueError("subset probs sum to 0")
        probs = probs / s
        out = np.zeros(self.n_arms, dtype=np.float64)
        out[idx] = probs
        return out

    def decay(self, factor: float) -> None:
        return None

    def print_summary(self) -> None:
        names = self._arm_names or [str(i) for i in range(self._n_arms)]
        post = self.posterior()

        # Create rich table
        table = Table(
            title="FixedSampler (fixed prior probs)",
            box=rich.box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
            width=120,  # Match display.py table width
        )

        # Add columns
        table.add_column("arm", style="white", width=28)
        table.add_column("base", justify="right", style="blue")
        table.add_column("prob", justify="right", style="bright_green")

        # Add rows
        for i, name in enumerate(names):
            # Split name by "/" and take last part, then last 28 chars
            if isinstance(name, str):
                display_name = name.split("/")[-1][-28:]
            else:
                display_name = str(name)
            table.add_row(
                display_name,
                f"{self._baseline[i]:.4f}",
                f"{post[i]:.4f}",
            )

        # Print directly to console
        console = Console()
        console.print(table)
