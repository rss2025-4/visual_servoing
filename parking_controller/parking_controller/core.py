import math
from typing import Protocol

import equinox as eqx
import jax
import optax
from beartype import beartype as typechecker
from jax import lax
from jax import numpy as jnp
from jax import tree_util as jtu
from jaxtyping import jaxtyped

from libracecar.plot import plot_ctx
from libracecar.specs import path, path_segment, position, turn_angle_limit
from libracecar.utils import debug_callback, debug_print, flatten_n_tree, fval, jit


class patheval(Protocol):
    def calc_loss(self, p: path) -> fval: ...


# class compute_score_(eqx.Module):
#     target: position

#     @jaxtyped(typechecker=typechecker)
#     def calc_loss(self, p: path) -> fval:
#         final_point, _ = p.move(position(jnp.array([0.0, 0.0]), jnp.array(0.0)))
#         target = self.target

#         loss = jnp.linalg.norm(target.coord - final_point.coord) ** 2
#         loss += (target.heading - final_point.heading) ** 2

#         loss += jnp.sum(p.map_parts(lambda x: jnp.abs(x.length))) / 100

#         return loss


class compute_score(eqx.Module):
    parking_distance: float = 0.75
    relative_x: float = 0.0
    relative_y: float = 0.0

    @jaxtyped(typechecker=typechecker)
    def calc_loss(self, p: path) -> fval:
        final_point, _ = p.move(position(jnp.array([0.0, 0.0]), jnp.array(0.0)))

        offset = (
            jnp.array([jnp.cos(final_point.heading), jnp.sin(final_point.heading)])
            * self.parking_distance
        )
        final_point_ahead = final_point.coord + offset

        x = final_point_ahead - jnp.array([self.relative_x, self.relative_y])
        loss = jnp.sum(x**2)

        def loss_len(l: fval, first: bool):
            ans = lax.select(l > 0, l, -l * 2.0)

            if first:
                return ans / 120
            else:
                return ans / 100

        for i, x in enumerate(p):
            loss += loss_len(x.length, i == 0)

        return loss


def loss_nan_cb(loss: fval, p: path):
    if jnp.isnan(loss):
        print("path caused nan loss:", p)
        print(p.move()[0])
        print(p.move()[1])


@jit
def gradient_descent_one(init: path, scoring: patheval = compute_score()) -> path:

    optim = optax.adamw(0.01)

    init_pos = position(jnp.array([0.0, 0.0]), jnp.array(0.0))

    @jax.jit
    def update(p: path, opt_state):
        loss, grads = jax.value_and_grad(scoring.calc_loss)(p)
        # debug_callback(loss_nan_cb, loss, p)
        updates, opt_state = optim.update(grads, opt_state, p)  # type: ignore
        p = eqx.apply_updates(p, updates)
        p = p.clip()
        return p, opt_state

    opt_state = optim.init(init)  # type: ignore

    ((ans, _), _) = lax.scan(
        lambda c, _: (update(c[0], c[1]), None),
        xs=None,
        init=(init, opt_state),
        length=1000,
    )

    # print(ans.move(init_pos).coord)
    # print(ans.move(init_pos).heading)
    # for x in ans.parts:
    #     print(x.length, x.angle)

    return ans


@jit
def gradient_descent_batched(scoring: patheval, ctx: plot_ctx) -> tuple[path, plot_ctx]:

    a_cands = jnp.linspace(-turn_angle_limit, turn_angle_limit, 5)
    l_cands = jnp.linspace(-math.pi / 5, math.pi / 5, 5)

    def make_path(a: fval, l: fval):
        seg = path_segment(a, l)
        return path.from_parts(*(seg for _ in range(5)))

    path_cands = jax.vmap(lambda l: jax.vmap(lambda a: make_path(a, l))(a_cands))(
        l_cands
    )
    path_cands = flatten_n_tree(path_cands, 2)

    ans = jax.vmap(lambda c: gradient_descent_one(c, scoring))(path_cands)

    # ctx += ans.plot()

    ans_losses = jax.vmap(scoring.calc_loss)(ans)
    idx = jnp.argmin(ans_losses)

    return jtu.tree_map(lambda x: x[idx], ans), ctx


# @eqx.filter_jit
@jit
def compute(scorer: patheval):
    ctx = plot_ctx.create(100)
    ans, ctx = gradient_descent_batched(scorer, ctx)
    ctx += ans.plot()
    ctx.check()
    return ans, ctx
