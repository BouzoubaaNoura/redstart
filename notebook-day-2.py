import marimo

__generated_with = "0.13.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Redstart: A Lightweight Reusable Booster""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(src="public/images/redstart.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Project Redstart is an attempt to design the control systems of a reusable booster during landing.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In principle, it is similar to SpaceX's Falcon Heavy Booster.

    >The Falcon Heavy booster is the first stage of SpaceX's powerful Falcon Heavy rocket, which consists of three modified Falcon 9 boosters strapped together. These boosters provide the massive thrust needed to lift heavy payloads—like satellites or spacecraft—into orbit. After launch, the two side boosters separate and land back on Earth for reuse, while the center booster either lands on a droneship or is discarded in high-energy missions.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.center(
        mo.Html("""
    <iframe width="560" height="315" src="https://www.youtube.com/embed/RYUr-5PYA7s?si=EXPnjNVnqmJSsIjc" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>""")
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Dependencies""")
    return


@app.cell(hide_code=True)
def _():
    import scipy
    import scipy.integrate as sci

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    from tqdm import tqdm

    # The use of autograd is optional in this project, but it may come in handy!
    import autograd
    import autograd.numpy as np
    import autograd.numpy.linalg as la
    from autograd import isinstance, tuple
    return FFMpegWriter, FuncAnimation, mpl, np, plt, scipy, tqdm


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## The Model

    The Redstart booster in model as a rigid tube of length $2 \ell$ and negligible diameter whose mass $M$ is uniformly spread along its length. It may be located in 2D space by the coordinates $(x, y)$ of its center of mass and the angle $\theta$ it makes with respect to the vertical (with the convention that $\theta > 0$ for a left tilt, i.e. the angle is measured counterclockwise)

    This booster has an orientable reactor at its base ; the force that it generates is of amplitude $f>0$ and the angle of the force with respect to the booster axis is $\phi$ (with a counterclockwise convention).

    We assume that the booster is subject to gravity, the reactor force and that the friction of the air is negligible.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.center(mo.image(src="public/images/geometry.svg"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Constants

    For the sake of simplicity (this is merely a toy model!) in the sequel we assume that: 

      - the total length $2 \ell$ of the booster is 2 meters,
      - its mass $M$ is 1 kg,
      - the gravity constant $g$ is 1 m/s^2.

    This set of values is not realistic, but will simplify our computations and do not impact the structure of the booster dynamics.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Helpers

    ### Rotation matrix

    $$ 
    \begin{bmatrix}
    \cos \alpha & - \sin \alpha \\
    \sin \alpha &  \cos \alpha  \\
    \end{bmatrix}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(np):
    def R(alpha):
        return np.array([
            [np.cos(alpha), -np.sin(alpha)], 
            [np.sin(alpha),  np.cos(alpha)]
        ])
    return (R,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Videos

    It will be very handy to make small videos to visualize the evolution of our booster!
    Here is an example of how such videos can be made with Matplotlib and displayed in marimo.
    """
    )
    return


@app.cell(hide_code=True)
def _(FFMpegWriter, FuncAnimation, mo, np, plt, tqdm):
    def make_video(output):
        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        num_frames = 100
        fps = 30 # Number of frames per second

        def animate(frame_index):    
            # Clear the canvas and redraw everything at each step
            plt.clf()
            plt.xlim(0, 2*np.pi)
            plt.ylim(-1.5, 1.5)
            plt.title(f"Sine Wave Animation - Frame {frame_index+1}/{num_frames}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid(True)

            x = np.linspace(0, 2*np.pi, 100)
            phase = frame_index / 10
            y = np.sin(x + phase)
            plt.plot(x, y, "r-", lw=2, label=f"sin(x + {phase:.1f})")
            plt.legend()

            pbar.update(1)

        pbar = tqdm(total=num_frames, desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=num_frames)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")

    _filename = "wave_animation.mp4"
    make_video(_filename)
    mo.show_code(mo.video(src=_filename))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Getting Started""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Constants

    Define the Python constants `g`, `M` and `l` that correspond to the gravity constant, the mass and half-length of the booster.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    g = 1.0
    M = 1.0
    l = 1
    return M, g, l


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Forces

    Compute the force $(f_x, f_y) \in \mathbb{R}^2$ applied to the booster by the reactor.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    \begin{align*}
    f_x & = -f \sin (\theta + \phi) \\
    f_y & = +f \cos(\theta +\phi)
    \end{align*}
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Center of Mass

    Give the ordinary differential equation that governs $(x, y)$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    \begin{align*}
    M \ddot{x} & = -f \sin (\theta + \phi) \\
    M \ddot{y} & = +f \cos(\theta +\phi) - Mg
    \end{align*}
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Moment of inertia

    Compute the moment of inertia $J$ of the booster and define the corresponding Python variable `J`.
    """
    )
    return


@app.cell
def _(M, l):
    J = M * l * l / 3
    J
    return (J,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Tilt

    Give the ordinary differential equation that governs the tilt angle $\theta$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$
    J \ddot{\theta} = - \ell (\sin \phi)  f
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Simulation

    Define a function `redstart_solve` that, given as input parameters: 

      - `t_span`: a pair of initial time `t_0` and final time `t_f`,
      - `y0`: the value of the state `[x, dx, y, dy, theta, dtheta]` at `t_0`,
      - `f_phi`: a function that given the current time `t` and current state value `y`
         returns the values of the inputs `f` and `phi` in an array.

    returns:

      - `sol`: a function that given a time `t` returns the value of the state `[x, dx, y, dy, theta, dtheta]` at time `t` (and that also accepts 1d-arrays of times for multiple state evaluations).

    A typical usage would be:

    ```python
    def free_fall_example():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi(t, y):
            return np.array([0.0, 0.0]) # input [f, phi]
        sol = redstart_solve(t_span, y0, f_phi)
        t = np.linspace(t_span[0], t_span[1], 1000)
        y_t = sol(t)[2]
        plt.plot(t, y_t, label=r"$y(t)$ (height in meters)")
        plt.plot(t, l * np.ones_like(t), color="grey", ls="--", label=r"$y=\ell$")
        plt.title("Free Fall")
        plt.xlabel("time $t$")
        plt.grid(True)
        plt.legend()
        return plt.gcf()
    free_fall_example()
    ```

    Test this typical example with your function `redstart_solve` and check that its graphical output makes sense.
    """
    )
    return


@app.cell(hide_code=True)
def _(J, M, g, l, np, scipy):
    def redstart_solve(t_span, y0, f_phi):
        def fun(t, state):
            x, dx, y, dy, theta, dtheta = state
            f, phi = f_phi(t, state)
            d2x = (-f * np.sin(theta + phi)) / M
            d2y = (+ f * np.cos(theta + phi)) / M - g
            d2theta = (- l * np.sin(phi)) * f / J
            return np.array([dx, d2x, dy, d2y, dtheta, d2theta])
        r = scipy.integrate.solve_ivp(fun, t_span, y0, dense_output=True)
        return r.sol
    return (redstart_solve,)


@app.cell(hide_code=True)
def _(l, np, plt, redstart_solve):
    def free_fall_example():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi(t, y):
            return np.array([0.0, 0.0]) # input [f, phi]
        sol = redstart_solve(t_span, y0, f_phi)
        t = np.linspace(t_span[0], t_span[1], 1000)
        y_t = sol(t)[2]
        plt.plot(t, y_t, label=r"$y(t)$ (height in meters)")
        plt.plot(t, l * np.ones_like(t), color="grey", ls="--", label=r"$y=\ell$")
        plt.title("Free Fall")
        plt.xlabel("time $t$")
        plt.grid(True)
        plt.legend()
        return plt.gcf()
    free_fall_example()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Controlled Landing

    Assume that $x$, $\dot{x}$, $\theta$ and $\dot{\theta}$ are null at $t=0$. For $y(0)= 10$ and $\dot{y}(0) = - 2*\ell$,  can you find a time-varying force $f(t)$ which, when applied in the booster axis ($\theta=0$), yields $y(5)=\ell$ and $\dot{y}(5)=0$?

    Simulate the corresponding scenario to check that your solution works as expected.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$
    % y(t)
    y(t)
    = \frac{2(5-\ell)}{125}\,t^3
      + \frac{3\ell-10}{25}\,t^2
      - 2\,t
      + 10
    $$

    $$
    % f(t)
    f(t)
    = M\!\Bigl[
        \frac{12(5-\ell)}{125}\,t
        + \frac{6\ell-20}{25}
        + g
      \Bigr].
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(M, g, l, np, plt, redstart_solve):

    def smooth_landing_example():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, -2.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi_smooth_landing(t, state):
            a = 12 * (5 - l) / 125
            b = (6 * l - 20) / 25
            return np.array([M * (a * t + b + g), 0])
        sol = redstart_solve(t_span, y0, f_phi=f_phi_smooth_landing)
        t = np.linspace(t_span[0], t_span[1], 1000)
        y_t = sol(t)[2]
        plt.plot(t, y_t, label=r"$y(t)$ (height in meters)")
        plt.plot(t, l * np.ones_like(t), color="grey", ls="--", label=r"$y=\ell$")
        plt.title("Controlled Landing")
        plt.xlabel("time $t$")
        plt.grid(True)
        plt.legend()
        return plt.gcf()
    smooth_landing_example()
    return


@app.cell
def _(M, g, l, np, plt, redstart_solve):
    def smooth_landing_example_force():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, -2.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi_smooth_landing(t, state):
            a = 12 * (5 - l) / 125
            b = (6 * l - 20) / 25
            return np.array([M * (a * t + b + g), 0])
        sol = redstart_solve(t_span, y0, f_phi=f_phi_smooth_landing)
        t = np.linspace(t_span[0], t_span[1], 1000)
        y_t = sol(t)[2]
        plt.plot(t, y_t, label=r"$y(t)$ (height in meters)")
        plt.plot(t, l * np.ones_like(t), color="grey", ls="--", label=r"$y=\ell$")
        plt.title("Controlled Landing")
        plt.xlabel("time $t$")
        plt.grid(True)
        plt.legend()
        return plt.gcf()
    smooth_landing_example_force()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Drawing

    Create a function that draws the body of the booster, the flame of its reactor as well as its target landing zone on the ground (of coordinates $(0, 0)$).

    The drawing can be very simple (a rectangle for the body and another one of a different color for the flame will do perfectly!).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.center(mo.image("public/images/booster_drawing.png"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Make sure that the orientation of the flame is correct and that its length is proportional to the force $f$ with the length equal to $\ell$ when $f=Mg$.

    The function shall accept the parameters `x`, `y`, `theta`, `f` and `phi`.
    """
    )
    return


@app.cell(hide_code=True)
def _(M, R, g, l, mo, mpl, np, plt):
    def draw_booster(x=0, y=l, theta=0.0, f=0.0, phi=0.0, axes=None, **options):
        L = 2 * l
        if axes is None:
            _fig, axes = plt.subplots()

        axes.set_facecolor('#F0F9FF') 

        ground = np.array([[-2*l, 0], [2*l, 0], [2*l, -l], [-2*l, -l], [-2*l, 0]]).T
        axes.fill(ground[0], ground[1], color="#E3A857", **options)

        b = np.array([
            [l/10, -l], 
            [l/10, l], 
            [0, l+l/10], 
            [-l/10, l], 
            [-l/10, -l], 
            [l/10, -l]
        ]).T
        b = R(theta) @ b
        axes.fill(b[0]+x, b[1]+y, color="black", **options)

        ratio = l / (M*g) # when f= +MG, the flame length is l 

        flame = np.array([
            [l/10, 0], 
            [l/10, - ratio * f], 
            [-l/10, - ratio * f], 
            [-l/10, 0], 
            [l/10, 0]
        ]).T
        flame = R(theta+phi) @ flame
        axes.fill(
            flame[0] + x + l * np.sin(theta), 
            flame[1] + y - l * np.cos(theta), 
            color="#FF4500", 
            **options
        )

        return axes

    _axes = draw_booster(x=0.0, y=20*l, theta=np.pi/8, f=M*g, phi=np.pi/8)
    _fig = _axes.figure
    _axes.set_xlim(-4*l, 4*l)
    _axes.set_ylim(-2*l, 24*l)
    _axes.set_aspect("equal")
    _axes.grid(True)
    _MaxNLocator = mpl.ticker.MaxNLocator
    _axes.xaxis.set_major_locator(_MaxNLocator(integer=True))
    _axes.yaxis.set_major_locator(_MaxNLocator(integer=True))
    _axes.set_axisbelow(True)
    mo.center(_fig)
    return (draw_booster,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Visualisation

    Produce a video of the booster for 5 seconds when

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=0$ and $\phi=0$

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=Mg$ and $\phi=0$

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=Mg$ and $\phi=\pi/8$

      - the parameters are those of the controlled landing studied above.

    As an intermediary step, you can begin the with production of image snapshots of the booster location (every 1 sec).
    """
    )
    return


@app.cell(hide_code=True)
def _(draw_booster, l, mo, np, plt, redstart_solve):
    def sim_1():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([0.0, 0.0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig, axes = plt.subplots(1, 6, sharey=True)
        num_snapshots = 6
        for i, t in enumerate(np.linspace(t_span[0], 5.0, num_snapshots)):
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            ax = draw_booster(x, y, theta, f, phi, axes=axes[i])
            ax.set_xticks([0.0])
            ax.set_xlim(-4*l, +4*l)
            ax.set_ylim(-2*l, +12*l)
            ax.set_aspect("equal")
            ax.set_xlabel(rf"$t={t}$")
            ax.set_axisbelow(True)
            ax.grid(True)
        w, h = fig.get_size_inches()
        fig.set_size_inches(12, h) 
        return mo.center(fig)

    sim_1()
    return


@app.cell(hide_code=True)
def _(M, draw_booster, g, l, mo, np, plt, redstart_solve):
    def sim_2():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([M*g, 0.0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig, axes = plt.subplots(1, 6, sharey=True)
        num_snapshots = 6
        for i, t in enumerate(np.linspace(t_span[0], 5.0, num_snapshots)):
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            ax = draw_booster(x, y, theta, f, phi, axes=axes[i])
            ax.set_xticks([0.0])
            ax.set_xlim(-4*l, +4*l)
            ax.set_ylim(-2*l, +12*l)
            ax.set_aspect("equal")
            ax.set_xlabel(rf"$t={t}$")
            ax.set_axisbelow(True)
            ax.grid(True)
        w, h = fig.get_size_inches()
        fig.set_size_inches(12, h) 
        return mo.center(fig)

    sim_2()
    return


@app.cell(hide_code=True)
def _(M, draw_booster, g, l, mo, np, plt, redstart_solve):
    def sim_3():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([M*g, np.pi/8])
        sol = redstart_solve(t_span, y0, f_phi)

        fig, axes = plt.subplots(1, 6, sharey=True)
        num_snapshots = 6
        for i, t in enumerate(np.linspace(t_span[0], 5.0, num_snapshots)):
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            ax = draw_booster(x, y, theta, f, phi, axes=axes[i])
            ax.set_xticks([0.0])
            ax.set_xlim(-4*l, +4*l)
            ax.set_ylim(-2*l, +12*l)
            ax.set_aspect("equal")
            ax.set_xlabel(rf"$t={t}$")
            ax.set_axisbelow(True)
            ax.grid(True)
        w, h = fig.get_size_inches()
        fig.set_size_inches(12, h) 
        return mo.center(fig)

    sim_3()
    return


@app.cell(hide_code=True)
def _(M, draw_booster, g, l, mo, np, plt, redstart_solve):
    def sim_4():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, -2.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi(t, state):
            a = 12 * (5 - l) / 125
            b = (6 * l - 20) / 25
            return np.array([M * (a * t + b + g), 0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig, axes = plt.subplots(1, 6, sharey=True)
        num_snapshots = 6
        for i, t in enumerate(np.linspace(t_span[0], 5.0, num_snapshots)):
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            ax = draw_booster(x, y, theta, f, phi, axes=axes[i])
            ax.set_xticks([0.0])
            ax.set_xlim(-4*l, +4*l)
            ax.set_ylim(-2*l, +12*l)
            ax.set_aspect("equal")
            ax.set_xlabel(rf"$t={t}$")
            ax.set_axisbelow(True)
            ax.grid(True)
        w, h = fig.get_size_inches()
        fig.set_size_inches(12, h) 
        return mo.center(fig)

    sim_4()
    return


@app.cell(hide_code=True)
def _(
    FFMpegWriter,
    FuncAnimation,
    draw_booster,
    l,
    mo,
    np,
    plt,
    redstart_solve,
    tqdm,
):
    def video_sim_1():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([0.0, 0.0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        axes = plt.gca()
        num_frames = 100
        fps = 30 # 30 frames per second
        ts = np.linspace(t_span[0], t_span[1], int(np.round(t_span[1] * fps)) + 1)
        output = "sim_1.mp4"

        def animate(t):    
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            axes.clear()
            draw_booster(x, y, theta, f, phi, axes=axes)
            axes.set_xticks([0.0])
            axes.set_xlim(-4*l, +4*l)
            axes.set_ylim(-2*l, +12*l)
            axes.set_aspect("equal")
            axes.set_xlabel(rf"$t={t:.1f}$")
            axes.set_axisbelow(True)
            axes.grid(True)

            pbar.update(1)

        pbar = tqdm(total=len(ts), desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=ts)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")
        return output

    mo.video(src=video_sim_1())
    return


@app.cell(hide_code=True)
def _(
    FFMpegWriter,
    FuncAnimation,
    M,
    draw_booster,
    g,
    l,
    mo,
    np,
    plt,
    redstart_solve,
    tqdm,
):
    def video_sim_2():
        L = 2*l

        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([M*g, 0.0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        axes = plt.gca()
        num_frames = 100
        fps = 30 # 30 frames per second
        ts = np.linspace(t_span[0], t_span[1], int(np.round(t_span[1] * fps)) + 1)
        output = "sim_2.mp4"

        def animate(t):    
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            axes.clear()
            draw_booster(x, y, theta, f, phi, axes=axes)
            axes.set_xticks([0.0])
            axes.set_xlim(-4*l, +4*l)
            axes.set_ylim(-2*l, +12*l)
            axes.set_aspect("equal")
            axes.set_xlabel(rf"$t={t:.1f}$")
            axes.set_axisbelow(True)
            axes.grid(True)

            pbar.update(1)

        pbar = tqdm(total=len(ts), desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=ts)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")
        return output

    mo.video(src=video_sim_2())
    return


@app.cell(hide_code=True)
def _(
    FFMpegWriter,
    FuncAnimation,
    M,
    draw_booster,
    g,
    l,
    mo,
    np,
    plt,
    redstart_solve,
    tqdm,
):
    def video_sim_3():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([M*g, np.pi/8])
        sol = redstart_solve(t_span, y0, f_phi)

        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        axes = plt.gca()
        num_frames = 100
        fps = 30 # 30 frames per second
        ts = np.linspace(t_span[0], t_span[1], int(np.round(t_span[1] * fps)) + 1)
        output = "sim_3.mp4"

        def animate(t):    
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            axes.clear()
            draw_booster(x, y, theta, f, phi, axes=axes)
            axes.set_xticks([0.0])
            axes.set_xlim(-4*l, +4*l)
            axes.set_ylim(-2*l, +12*l)
            axes.set_aspect("equal")
            axes.set_xlabel(rf"$t={t:.1f}$")
            axes.set_axisbelow(True)
            axes.grid(True)

            pbar.update(1)

        pbar = tqdm(total=len(ts), desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=ts)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")
        return output

    mo.video(src=video_sim_3())
    return


@app.cell(hide_code=True)
def _(
    FFMpegWriter,
    FuncAnimation,
    M,
    draw_booster,
    g,
    l,
    mo,
    np,
    plt,
    redstart_solve,
    tqdm,
):
    def video_sim_4():
        L = 2*l
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, -2.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi(t, state):
            a = 12 * (5 - l) / 125
            b = (6 * l - 20) / 25
            return np.array([M * (a * t + b + g), 0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        axes = plt.gca()
        num_frames = 100
        fps = 30 # 30 frames per second
        ts = np.linspace(t_span[0], t_span[1], int(np.round(t_span[1] * fps)) + 1)
        output = "sim_4.mp4"

        def animate(t):    
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            axes.clear()
            draw_booster(x, y, theta, f, phi, axes=axes)
            axes.set_xticks([0.0])
            axes.set_xlim(-4*l, +4*l)
            axes.set_ylim(-2*l, +12*l)
            axes.set_aspect("equal")
            axes.set_xlabel(rf"$t={t:.1f}$")
            axes.set_axisbelow(True)
            axes.grid(True)

            pbar.update(1)

        pbar = tqdm(total=len(ts), desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=ts)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")
        return output

    mo.video(src=video_sim_4())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Linearized Dynamics""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Equilibria

    We assume that $|\theta| < \pi/2$, $|\phi| < \pi/2$ and that $f > 0$. What are the possible equilibria of the system for constant inputs $f$ and $\phi$ and what are the corresponding values of these inputs?
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Nous cherchons les états d'équilibre du système sous les hypothèses suivantes :

    - \( |\theta| < \frac{\pi}{2} \)
    - \( |\phi| < \frac{\pi}{2} \)
    - \( f > 0 \)
    - \( f \), \( \phi \) constants

    ---
    On rappel les équations suivantes:

    1. **Mouvement horizontal** :

    \[  
    \ddot{x} = -f \sin(\theta + \phi)
    \]

    3. **Mouvement vertical** :

    \[
    M \ddot{y} = f \cos(\theta + \phi) - Mg
    \]

    5. **Rotation** :

    \[
    J \ddot{\theta} = -l f \sin(\phi)
    \]

    ---

     Conditions d'équilibre (toutes les dérivées nulles) :

    \[
    \ddot{x} = \ddot{y} = \ddot{\theta} = 0
    \]

    ---

     Résolution :

    1. \(\ddot{\theta} = 0 \Rightarrow \sin(\phi) = 0 \Rightarrow \phi = 0\)

    2. \(\ddot{x} = 0 \Rightarrow \sin(\theta + \phi) = 0 \Rightarrow \theta = 0\)

    3. \(\ddot{y} = 0 \Rightarrow f \cos(\theta + \phi) = Mg \Rightarrow f = Mg\)

    ---

     État d'équilibre :

    \[
    (x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (\text{libre}, 0, \text{libre}, 0, 0, 0)
    \]

    \[
    f = Mg, \quad \phi = 0
    \]
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Linearized Model

    Introduce the error variables $\Delta x$, $\Delta y$, $\Delta \theta$, and $\Delta f$ and $\Delta \phi$ of the state and input values with respect to the generic equilibrium configuration.
    What are the linear ordinary differential equations that govern (approximately) these variables in a neighbourhood of the equilibrium?
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Linéarisation du modèle autour de l'équilibre

    1. Variables de perturbation

    On introduit les déviations autour de l’équilibre :

    $$
    \begin{aligned}
    x &= x_{\text{eq}} + \Delta x \\
    y &= y_{\text{eq}} + \Delta y \\
    \theta &= 0 + \Delta \theta \\
    f &= Mg + \Delta f \\
    \phi &= 0 + \Delta \phi
    \end{aligned}
    $$

    #### 2. Approximation de Taylor (petits angles)

    $$
    \begin{aligned}
    \sin(\theta + \phi) &\approx \theta + \phi \\
    \cos(\theta + \phi) &\approx 1 \\
    \sin(\phi) &\approx \phi
    \end{aligned}
    $$

    #### 3. Équations du mouvement linéarisées

    $$
    \begin{cases}
    M \ddot{\Delta x} = -(Mg+\Delta f)(\Delta \theta + \Delta \phi) \\
    M \ddot{\Delta y} = \Delta f \\
    J \ddot{\Delta \theta} = -l Mg \Delta \phi
    \end{cases}
    $$

    Dans le voisinage de l'équilibre, on néglige le produit $(\Delta f \cdot \theta)$ , ainsi on trouve: 

    $$
    \begin{cases}
     \ddot{\Delta x} = -g(\Delta \theta + \Delta \phi) \\
    M \ddot{\Delta y} = \Delta f \\
    J \ddot{\Delta \theta} = -l Mg \Delta \phi
    \end{cases}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Standard Form

    What are the matrices $A$ and $B$ associated to this linear model in standard form?
    Define the corresponding NumPy arrays `A` and `B`.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Forme Standard 

    Pour obtenir la forme standard \( \dot{X} = A X + B U \) du modèle linéarisé, on définit :

    ---


    $$
    X = \begin{bmatrix}
    \Delta x \\
    \Delta \dot{x} \\
    \Delta y \\
    \Delta \dot{y} \\
    \Delta \theta \\
    \Delta \dot{\theta}
    \end{bmatrix}, \quad
    U = \begin{bmatrix}
    \Delta f \\
    \Delta \varphi
    \end{bmatrix}
    $$

    ##  Équations linéarisées

    $$
    \begin{aligned}
    \Delta \ddot{x} &= -g \Delta \theta - g \Delta \phi \\
    \Delta \ddot{y} &= \frac{1}{M} \Delta f \\
    \Delta \ddot{\theta} &= -\frac{3g}{\ell} \Delta \phi
    \end{aligned}
    $$

    ---

    ## Matrix A

    \[
    A = \begin{bmatrix}
    0 & 1 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & -g & 0 \\
    0 & 0 & 0 & 1 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 1 \\
    0 & 0 & 0 & 0 & 0 & 0
    \end{bmatrix}
    \]

    ---

    ## Matrix B

    \[
    B = \begin{bmatrix}
    0 & 0 \\
    0 & -g \\
    0 & 0 \\
    \frac{1}{M} & 0 \\
    0 & 0 \\
    0 & -\frac{3g}{\ell}
    \end{bmatrix}
    \]
    """
    )
    return


@app.cell
def _(M, g, l, np):
    # Matrice A
    A = np.array([
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, -g, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0]
    ])

    # Matrice B
    B = np.array([
        [0, 0],
        [0, -g],
        [0, 0],
        [1/M, 0],
        [0, 0],
        [0, -3*g/l]
    ])
    return A, B


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Stability

    Is the generic equilibrium asymptotically stable?
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Analyse de Stabilité du Système Linéarisé

    Pour déterminer si l'équilibre \((\theta = 0, \phi = 0, f = Mg)\) est asymptotiquement stable, nous analysons les valeurs propres de la matrice \(A\) du système linéarisé.

    Matrice \(A\) du système linéarisé :

    \[
    A = \begin{bmatrix}
    0 & 1 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & -g & 0 \\
    0 & 0 & 0 & 1 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 1 \\
    0 & 0 & 0 & 0 & 0 & 0
    \end{bmatrix}
    \]


    Les valeurs propres \(\lambda\) sont solutions de \(\det(A - \lambda I) = 0\).  
    On obtient :

    \[
    \lambda^6 = 0 \Rightarrow \lambda \in \{0, 0, 0, 0, 0, 0\}
    \]

    ---



    Toutes les valeurs propres sont nulles. Alors, le système est : **Non asymptotiquement stable**.

    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Controllability

    Is the linearized model controllable?
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    On pose C = [B \ AB \ A^2B \ A^3B \ A^4B \ A^5B] 

    Après calcul (par exemple avec NumPy), on trouve :  
    \[
    \text{rang}(\mathcal{C}) = 6
    \]  

    Donc, d'après Kalman Criterion Le système est  contrôlable.
    """
    )
    return


@app.cell
def _(A, B):
    from numpy import shape, column_stack
    from numpy.linalg import matrix_power
    from numpy.linalg import matrix_rank


    def KCM(A, B):
        n = shape(A)[0]  
        return column_stack([matrix_power(A, k) @ B for k in range(n)])
    rank_C = matrix_rank(KCM(A, B))
    print("Rang de C :", rank_C)
    return KCM, matrix_rank


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Lateral Dynamics

    We limit our interest in the lateral position $x$, the tilt $\theta$ and their derivatives (we are for the moment fine with letting $y$ and $\dot{y}$ be uncontrolled). We also set $f = M g$ and control the system only with $\phi$.

    What are the new (reduced) matrices $A$ and $B$ for this reduced system?
    Check the controllability of this new system.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Nous nous intéressons uniquement à la dynamique latérale du système, c’est-à-dire la position \( x \), l’angle d’inclinaison \( \theta \), et leurs dérivées.  
    On suppose que :

    - \( f = Mg \) (constante),
    - Seul \( \phi \) est contrôlable.

    ---

    ## Matrices réduites du système

    La dynamique réduite s’écrit avec les variables d’état :

    \[
    X = \begin{bmatrix}
    \Delta x \\
    \Delta \dot{x} \\
    \Delta \theta \\
    \Delta \dot{\theta}
    \end{bmatrix},
    \quad
    U = \begin{bmatrix}
    \Delta \phi
    \end{bmatrix}
    \]

    \[
    \dot{X} = A X + B U
    \]

    ```python
    import numpy as np
    from numpy.linalg import matrix_rank

    g = 1      
    l = 1      

    # Matrice A réduite
    A_red = np.array([
        [0, 1, 0, 0],
        [0, 0, -g, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])

    # Matrice B réduite (contrôle uniquement via phi)
    B_red = np.array([
        [0],
        [-g],
        [0],
        [-3 * g / l]
    ])

    # Matrice de contrôlabilité
    def KCM(A, B):
        return np.hstack([B, A @ B, A @ A @ B, A @ A @ A @ B])

    C_red = KCM(A_red, B_red)
    rank_red = matrix_rank(C_red)

    print("Rang de C_red :", rank_red)
    ```

    ---

     **Rang de la matrice de contrôlabilité :** 4  
     Comme le rang est maximal (égal à la dimension de l’espace d’état),  **le système réduit est contrôlable**.

    """
    )
    return


@app.cell
def _(KCM, g, l, matrix_rank, np):
    # Matrices réduites

    A_red = np.array([
        [0, 1, 0, 0],
        [0, 0, -g, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])

    B_red = np.array([
        [0],
        [-g],
        [0],
        [-3*g/l]
    ])



    C_red = KCM(A_red, B_red)
    rank_red = matrix_rank(C_red)

    print("Rang de C_red :", rank_red)
    return A_red, B_red


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Linear Model in Free Fall

    Make graphs of $y(t)$ and $\theta(t)$ for the linearized model when $\phi(t)=0$,
    $x(0)=0$, $\dot{x}(0)=0$, $\theta(0) = 45 / 180  \times \pi$  and $\dot{\theta}(0) =0$. What do you see? How do you explain it?
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Simulation et tracé de y(t) et θ(t) pour le modèle linéarisé

    Nous considérons le modèle linéarisé complet suivant :

    \[
    \begin{aligned}
    \Delta \ddot{x} &= -g (\Delta \theta + \Delta \phi) \\
    \Delta \ddot{y} &= \frac{1}{M} \Delta f \\
    \Delta \ddot{\theta} &= -\frac{3g}{\ell} \Delta \phi
    \end{aligned}
    \]

    Sous les conditions :

    - \( \phi(t) = 0 \)
    - \( f(t) = Mg \Rightarrow \Delta f = 0 \)

    ---

    ## Simulation de \( x(t) \) et \( \theta(t) \)

    ```python

    from scipy.integrate import solve_ivp
    def linearized_system(t, Z):
        x, xdot, y, ydot, theta, thetadot = Z
        dxdt = xdot
        dxdotdt = -g * theta  
        dydt = ydot
        dydotdt = 0           
        dthetadt = thetadot
        dthetadotdt = 0       
        return [dxdt, dxdotdt, dydt, dydotdt, dthetadt, dthetadotdt]


    theta0 = np.pi / 4
    Z0 = [0, 0, 0, 0, theta0, 0]  


    t_span = (0, 10)
    t_eval = np.linspace(*t_span, 500)
    sol = solve_ivp(linearized_system, t_span, Z0, t_eval=t_eval)


    t = sol.t
    x = sol.y[0]
    theta = sol.y[4]


    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(t, x, label="$x(t)$", color="blue")
    plt.xlabel("Temps [s]")
    plt.ylabel("Position latérale x(t)")
    plt.title("Évolution de x(t)")
    plt.grid()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(t, theta, label=r"$\theta(t)$", color="orange")
    plt.xlabel("Temps [s]")
    plt.ylabel("Angle d'inclinaison θ(t) [rad]")
    plt.title("Évolution de θ(t)")
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()
    ```

    ## Interprétation

     1. Évolution de \( \theta(t) \)

    \[
    \theta(t) = \theta(0) = \frac{\pi}{4}
    \]

    reste **constante** dans le temps.

     vu que;

    - Le système est linéarisé autour de \( \phi = 0 \)
    - Et on impose \( \phi(t) = 0 \Rightarrow \Delta \phi = 0 \)

    D’où :

    \[
    \Delta \ddot{\theta} = -\frac{3g}{\ell} \Delta \phi = 0 \quad \Rightarrow \quad \dot{\theta} = \text{constante} \quad \Rightarrow \quad \theta = \text{constante}
    \]

    2. Évolution de \( x(t) \)

    L’accélération \( \ddot{x} \) dépend directement de l’inclinaison \( \theta \) :

    \[
    \Delta \ddot{x} = -g (\Delta \theta + \Delta \phi) = -g\theta \quad \text{(car } \phi = 0\text{)}
    \]

    Puisque \( \theta \) est **constant**, l’accélération \( \ddot{x}(t) \) est **constante** :

    \[
    \ddot{x}(t) = -g \theta_0 \quad \Rightarrow \quad x(t) = -\frac{1}{2} g \theta_0 t^2
    \]

    Donc le mobile **accélère latéralement** de manière **quadratique**, à cause de l’inclinaison initiale de la nacelle.


    Pour conclure;

    Le système est initialement incliné \( (\theta_0 > 0) \), mais sans contrôle actif \( (\phi = 0) \), donc :

    - La nacelle **penche** mais ne se **redresse pas**
    - L’inclinaison \( \theta(t) \) reste figée
    - La plateforme glisse sous l’effet de la gravité, d’où \( x(t) \) croît en valeur absolue avec le temps



    """
    )
    return


@app.cell
def _(g, np, plt):

    from scipy.integrate import solve_ivp

    def linearized_system(t, Z):
        x, xdot, y, ydot, theta, thetadot = Z
        dxdt = xdot
        dxdotdt = -g * theta  
        dydt = ydot
        dydotdt = 0           
        dthetadt = thetadot
        dthetadotdt = 0       
        return [dxdt, dxdotdt, dydt, dydotdt, dthetadt, dthetadotdt]


    theta0 = np.pi / 4
    Z0 = [0, 0, 0, 0, theta0, 0]  


    t_span = (0, 10)
    t_eval = np.linspace(*t_span, 500)
    sol = solve_ivp(linearized_system, t_span, Z0, t_eval=t_eval)


    t = sol.t
    x = sol.y[0]
    theta = sol.y[4]


    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(t, x, label="$x(t)$", color="blue")
    plt.xlabel("Temps [s]")
    plt.ylabel("Position latérale x(t)")
    plt.title("Évolution de x(t)")
    plt.grid()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(t, theta, label=r"$\theta(t)$", color="orange")
    plt.xlabel("Temps [s]")
    plt.ylabel("Angle d'inclinaison θ(t) [rad]")
    plt.title("Évolution de θ(t)")
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()

    return solve_ivp, t_eval


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Manually Tuned Controller

    Try to find the two missing coefficients of the matrix 

    $$
    K =
    \begin{bmatrix}
    0 & 0 & ? & ?
    \end{bmatrix}
    \in \mathbb{R}^{4\times 1}
    $$ 

    such that the control law 

    $$
    \Delta \phi(t)
    = 
    - K \cdot
    \begin{bmatrix}
    \Delta x(t) \\
    \Delta \dot{x}(t) \\
    \Delta \theta(t) \\
    \Delta \dot{\theta}(t)
    \end{bmatrix} \in \mathbb{R}
    $$

    manages  when
    $\Delta x(0)=0$, $\Delta \dot{x}(0)=0$, $\Delta \theta(0) = 45 / 180  \times \pi$  and $\Delta \dot{\theta}(0) =0$ to: 

      - make $\Delta \theta(t) \to 0$ in approximately $20$ sec (or less),
      - $|\Delta \theta(t)| < \pi/2$ and $|\Delta \phi(t)| < \pi/2$ at all times,
      - (but we don't care about a possible drift of $\Delta x(t)$).

    Explain your thought process, show your iterations!

    Is your closed-loop model asymptotically stable?
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r""" """)
    return


@app.cell
def _(A_red, B_red, np, plt, scipy):
    def solve_linear_manual(t_span, y0, A_cl):
        def linear_dynamics(t, y):
            return A_cl @ y
        sol = scipy.integrate.solve_ivp(linear_dynamics, t_span, y0, dense_output=True)
        return sol.sol


    K_manual = np.array([0, 0, -1/75, -2/15]) 


    delta_zlat0 = np.array([0.0, 0.0, np.pi/4, 0.0])
    t_span_manual = [0.0, 30.0]
    t_manual = np.linspace(t_span_manual[0], t_span_manual[1], 300)


    A_cl_manual = A_red - B_red @ K_manual.reshape(1, -1)

    sol_manual = solve_linear_manual(t_span_manual, delta_zlat0, A_cl_manual)
    delta_zlat_t = sol_manual(t_manual)

    delta_theta_t = delta_zlat_t[2]
    delta_dtheta_t = delta_zlat_t[3]
    delta_phi_t = - K_manual @ delta_zlat_t

    fig_manual_final, axes_manual_final = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes_manual_final[0].plot(t_manual, delta_theta_t, label=r"$\Delta\theta(t)$ (rad)")
    axes_manual_final[0].plot(t_manual, np.zeros_like(t_manual), color="grey", ls="--")
    axes_manual_final[0].plot(t_manual, np.full_like(t_manual, np.pi/2), color="red", ls=":", label=r"$+\pi/2$")
    axes_manual_final[0].plot(t_manual, np.full_like(t_manual, -np.pi/2), color="red", ls=":", label=r"$-\pi/2$")




    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Controller Tuned with Pole Assignment

    Using pole assignement, find a matrix

    $$
    K_{pp} =
    \begin{bmatrix}
    ? & ? & ? & ?
    \end{bmatrix}
    \in \mathbb{R}^{4\times 1}
    $$ 

    such that the control law 

    $$
    \Delta \phi(t)
    = 
    - K_{pp} \cdot
    \begin{bmatrix}
    \Delta x(t) \\
    \Delta \dot{x}(t) \\
    \Delta \theta(t) \\
    \Delta \dot{\theta}(t)
    \end{bmatrix} \in \mathbb{R}
    $$

    satisfies the conditions defined for the manually tuned controller and additionally:

      - result in an asymptotically stable closed-loop dynamics,

      - make $\Delta x(t) \to 0$ in approximately $20$ sec (or less).

    Explain how you find the proper design parameters!
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Le système linéarisé est :

    \[
    \dot{X} = A X + B \Delta \phi, \quad X = 
    \begin{bmatrix}
    \Delta x \\
    \Delta \dot{x} \\
    \Delta \theta \\
    \Delta \dot{\theta}
    \end{bmatrix}
    \]

    avec :

    \[
    A = 
    \begin{bmatrix} 
    0 & 1 & 0 & 0 \\ 
    0 & 0 & -g & 0 \\ 
    0 & 0 & 0 & 1 \\ 
    0 & 0 & 0 & 0 
    \end{bmatrix}, \quad 
    B = 
    \begin{bmatrix} 
    0 \\ 
    -g \\ 
    0 \\ 
    -\frac{3g}{\ell}
    \end{bmatrix}
    \]

     Placement des pôles

    Le temps de réponse est approximativement :

    \[
    \text{Temps de réponse} \approx \frac{4}{|\text{Re(pôle)}|}
    \]
    Donc, pour un pôle à \(-0{,}2\), on a :
    \[
    \frac{4}{0.2} = 20 \text{ secondes}
    \]

    Nous choisissons les pôles :
    - Deux pôles à \(-0.2\) (pôle double) pour accélérer la convergence sans oscillation,
    - Un à \(-0.3\),
    - Un à \(-0.4\),

    Le polynôme caractéristique désiré est donc :

    \[
    (s + 0.2)^2 (s + 0.3) (s + 0.4) = s^4 + 1.1s^3 + 0.46s^2 + 0.076s + 0.0048
    \]

     Calcul de \( K_{pp} \)

    La dynamique en boucle fermée devient :

    \[
    \dot{X} = (A - B K_{pp}) X
    \]

    On veut :

    \[
    \text{det}(sI - (A - B K_{pp})) = s^4 + 1.1s^3 + 0.46s^2 + 0.076s + 0.0048
    \]

    En résolvant, on obtient :

    \[
    K_{pp} = 
    \begin{bmatrix}
    -0.0012 & -0.012 & 0.0008 & 0.0096
    \end{bmatrix}
    \]
    """
    )
    return


@app.cell
def _(g, l, np):
    from scipy.signal import place_poles

    A1 = np.array([
        [0, 1, 0, 0],
        [0, 0, -g, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])
    B1 = np.array([[0], [-g], [0], [-3*g/l]])

    # Pôles désirés
    poles = [-0.19, -0.2, -0.3, -0.4]  # Pôles distincts mais proches de -0.2


    # Placement de pôles
    K_pp = place_poles(A1, B1, poles).gain_matrix
    print("K_pp =", K_pp)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Controller Tuned with Optimal Control

    Using optimal, find a gain matrix $K_{oc}$ that satisfies the same set of requirements that the one defined using pole placement.

    Explain how you find the proper design parameters!
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r""" """)
    return


@app.cell
def _(g, l, np, plt, solve_ivp, t_eval):
    from scipy.linalg import solve_continuous_are

    # System matrices (g=1, l=1)
    A3 = np.array([
        [0, 1, 0, 0],
        [0, 0, -g, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])
    B3 = np.array([[0], [-g], [0], [-3*g/l]])

    # LQR weights
    Q = np.diag([1, 1, 10, 10])  # Prioritize angle stabilization
    R1 = 1.0  # Scalar

    # Solve Riccati equation
    P = solve_continuous_are(A3, B3, Q, R1)

    # Compute optimal gain
    K_oc = B3.T @ P  # Or (1/R) * B.T @ P
    print("Optimal gain K_oc =", K_oc)
    def closed_loop(t, x):
        u = -K_oc @ x
        return (A3 @ x + B3.flatten() * u).flatten()

    # Conditions initiales : Δx, Δẋ, Δθ, Δθ̇
    x0 = np.array([0, 0, np.pi/4, 0])

    # Intégration
    t_span2 = (0, 25)
    t_eval3 = np.linspace(*t_span2, 500)
    sol3 = solve_ivp(closed_loop, t_span2, x0, t_eval=t_eval3)

    # Extraire Δθ(t)
    theta3 = sol3.y[2, :]  # index 2 = Δθ

    # Tracer Δθ(t)
    plt.figure(figsize=(10, 5))
    plt.plot(t_eval, theta3, label=r'$\Delta \theta(t)$', color='blue')
    plt.axhline(np.pi/2, color='red', linestyle='--', alpha=0.3)
    plt.axhline(-np.pi/2, color='red', linestyle='--', alpha=0.3)
    plt.xlabel('Temps (s)')
    plt.ylabel(r'$\Delta \theta(t)$')
    plt.title('Évolution de Δθ(t) avec commande LQR')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Validation

    Test the two control strategies (pole placement and optimal control) on the "true" (nonlinear) model and check that they achieve their goal. Otherwise, go back to the drawing board and tweak the design parameters until they do!
    """
    )
    return


@app.cell
def _(J, M, g, l, np, plt, solve_ivp):




    K_pp1 = np.array([0.00152, 0.02446667, -0.14417333, -0.37148889])  
    K_oc1 = np.array([1.0, 3.75005266, -6.53144749, -5.02403229])      


    def nonlinear_model(t, X, K):
        x, x_dot, theta, theta_dot = X
        Delta_phi = -K @ X
        phi = Delta_phi  # Petit angle
    
        f = M * g  # Force constante
        x_ddot = -f * np.sin(theta + phi) / M
        y_ddot = f * np.cos(theta + phi) / M - g
        theta_ddot = -l * f * np.sin(phi) / J
    
        return [x_dot, x_ddot, theta_dot, theta_ddot]

    # Conditions initiales
    X0 = [0, 0, np.pi/4, 0]  # Δθ(0) = 45°


    t_span4 = [0, 20]
    t_eval4 = np.linspace(t_span4[0], t_span4[1], 1000)


    sol_pp = solve_ivp(nonlinear_model, t_span4, X0, args=(K_pp1,), t_eval=t_eval4)

    sol_oc = solve_ivp(nonlinear_model, t_span4, X0, args=(K_oc1,), t_eval=t_eval4)


    phi_pp = -K_pp1 @ sol_pp.y
    phi_oc = -K_oc1 @ sol_oc.y

    # Tracés
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(sol_pp.t, sol_pp.y[2], label='Pole Assignment')
    plt.plot(sol_oc.t, sol_oc.y[2], label='optimal control')
    plt.axhline(y=np.pi/2, color='r', linestyle='--', label='Limite π/2')
    plt.xlabel('Temps (s)')
    plt.ylabel('Δθ (rad)')
    plt.title('Angle du pendule')
    plt.legend()
    plt.grid()


    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
