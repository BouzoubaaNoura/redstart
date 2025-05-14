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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Dependencies""")
    return


@app.cell
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
    return FFMpegWriter, FuncAnimation, np, plt, tqdm


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


@app.cell
def _(np):
    def R(alpha):
        return np.array([
            [np.cos(alpha), -np.sin(alpha)], 
            [np.sin(alpha),  np.cos(alpha)]
        ])
    return


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


@app.cell
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
    (mo.video(src=_filename))
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


@app.cell
def _():
    l=1
    M=1
    g=1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Forces

    Compute the force $(f_x, f_y) \in \mathbb{R}^2$ applied to the booster by the reactor.
    """
    )
    return


@app.cell
def _(mo):
    mo.center(mo.image(src="public/images/schema simplificatif.png"))
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    En projetant la force \( f \) selon les angles \( \theta \) et \( \varphi \), on obtient :

    $$
    f_x = f \cdot \sin(\theta + \varphi)
    $$

    $$
    f_y = f \cdot \cos(\theta + \varphi)
    $$

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


@app.cell
def _(mo):
    mo.md(
        r"""
    En appliquant la **deuxième loi de Newton** au propulseur, on obtient :

    $$
    M \cdot \frac{d\vec{v}}{dt} = \vec{F}_{\text{réacteur}} + \vec{P}
    $$

    où :  
    - \( \vec{F}_{\text{réacteur}} \) est la force de poussée générée par le réacteur,  
    - \( \vec{P} \) représente le poids du propulseur.

    ---

    En projetant cette équation sur le repère global \( (x, y) \), on obtient :

    $$
    \begin{cases}
    M \cdot \ddot{x}(t) = f_x \\
    M \cdot \ddot{y}(t) = f_y - M \cdot g
    \end{cases}
    $$

    avec :  
    - \( \ddot{x}(t) \) et \( \ddot{y}(t) \) les composantes de l'accélération du centre de masse,  
    - \( f_x \) et \( f_y \) les composantes de la force du réacteur dans le repère global,  
    - \( M = 1\, \text{kg} \) la masse du propulseur,  
    - \( g = 1\, \text{m/s}^2 \) l'accélération gravitationnelle.

    ---

    D'après la question précédente, les composantes de la force s’écrivent :

    $$
    \begin{cases}
    f_x = f \cdot \sin(\theta + \varphi) \\
    f_y = f \cdot \cos(\theta + \varphi)
    \end{cases}
    $$

    ---

    En remplaçant dans le système précédent, et en utilisant \( M = 1 \) et \( g = 1 \), on obtient les **équations différentielles ordinaires (EDO)** qui régissent la position \( (x(t), y(t)) \) du centre de masse :

    $$
    \begin{cases}
    \ddot{x}(t) = f \cdot \sin(\theta + \varphi) \\
    \ddot{y}(t) = f \cdot \cos(\theta + \varphi) - 1
    \end{cases}
    $$

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
def _(mo):
    mo.md(
        r"""
    D'après les hypothèses d'uniformité de masse et de rigidité du tube, on obtient :

    $$
    J = \frac{1}{12} M \cdot L^2
    $$

    Ainsi on trouve:

    $$
    J = \frac{1}{3} M \cdot l^2
    $$

    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Tilt

    Give the ordinary differential equation that governs the tilt angle $\theta$.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    D'après le **principe fondamental de la dynamique**, on a :

    $$
    J \cdot \ddot{\theta} = \sigma
    $$

    Or :

    $$
    \sigma = l \cdot f \cdot \sin(\varphi) \\
    J = \frac{1}{3} M \cdot l^2
    $$

    Ainsi, on obtient :

    $$
    \ddot{\theta} = 3\cdot\frac{f \cdot \sin(\varphi)}{ M \cdot l}
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


@app.cell
def _(mo):
    mo.md(
        r"""

    ```python

    from scipy.integrate import solve_ivp

    def redstart_solve(t_span, y0, f_phi, M=1.0, l=1.0, g=1):
    
        def dynamics(t, y):
        
            x, dx, y, dy, theta, dtheta = y
            f, phi = f_phi(t, y)

            # Compute accelerations
            ddx = (f/M) * np.sin(theta + phi)
            ddy = (f/M) * np.cos(theta + phi) - g
            ddtheta = (3 * f * np.sin(phi)) / (M * l)

            return [dx, ddx, dy, ddy, dtheta, ddtheta]

        # Solve the ODE system
        solution = solve_ivp(dynamics, t_span, y0,
                            t_eval=np.linspace(t_span[0], t_span[1], 1000),
                            dense_output=True,
                            rtol=1e-6, atol=1e-9)
        return solution.sol  
    ```
    On appliquant la fonction redstart_solve sur l'example de freefall function: 

    ```python
    def free_fall_example():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]  # [x, dx, y, dy, theta, dtheta]
        def f_phi(t, y):
            return np.array([0.0, 0.0])  # No thrust, no gimbal
        sol = redstart_solve(t_span, y0, f_phi)
        t = np.linspace(t_span[0], t_span[1], 1000)
        y_t = sol(t)[2]  # Extract height (y) over time
        plt.plot(t, y_t, label=r"$y(t)$ (height in meters)")
        plt.plot(t, 1.0 * np.ones_like(t), color="grey", ls="--", label=r"$y=\ell$")
        plt.title("Free Fall of Redstart Booster")
        plt.xlabel("Time $t$ (seconds)")
        plt.ylabel("Height $y$ (meters)")
        plt.grid(True)
        plt.legend()
        plt.show()

    free_fall_example() 
    ```
    On trouve:

    """
    )
    return


@app.cell
def _(np, plt):
    from scipy.integrate import solve_ivp

    def redstart_solve(t_span, y0, f_phi, M=1.0, l=1.0, g=1):

        def dynamics(t, y):

            x, dx, y, dy, theta, dtheta = y
            f, phi = f_phi(t, y)

            # Compute accelerations
            ddx = (f/M) * np.sin(theta + phi)
            ddy = (f/M) * np.cos(theta + phi) - g
            ddtheta = (3 * f * np.sin(phi)) / (M * l)

            return [dx, ddx, dy, ddy, dtheta, ddtheta]

        # Solve the ODE system
        solution = solve_ivp(dynamics, t_span, y0,
                            t_eval=np.linspace(t_span[0], t_span[1], 1000),
                            dense_output=True,
                            rtol=1e-6, atol=1e-9)
        return solution.sol

    def free_fall_example():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]  # [x, dx, y, dy, theta, dtheta]
        def f_phi(t, y):
            return np.array([0.0, 0.0])  # No thrust, no gimbal
        sol = redstart_solve(t_span, y0, f_phi)
        t = np.linspace(t_span[0], t_span[1], 1000)
        y_t = sol(t)[2]  # Extract height (y) over time
        plt.plot(t, y_t, label=r"$y(t)$ (height in meters)")
        plt.plot(t, 1.0 * np.ones_like(t), color="grey", ls="--", label=r"$y=\ell$")
        plt.title("Free Fall of Redstart Booster")
        plt.xlabel("Time $t$ (seconds)")
        plt.ylabel("Height $y$ (meters)")
        plt.grid(True)
        plt.legend()
        plt.show()

    free_fall_example()

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Controlled Landing

    Assume that $x$, $\dot{x}$, $\theta$ and $\dot{\theta}$ are null at $t=0$. For $y(0)= 10$ and $\dot{y}(0)$, can you find a time-varying force $f(t)$ which, when applied in the booster axis ($\theta=0$), yields $y(5)=\ell$ and $\dot{y}(5)=0$?

    Simulate the corresponding scenario to check that your solution works as expected.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    D'après les résultats précédents, on a trouvé que :

    $$
    \ddot{y} = \frac{f}{M} \cdot \cos(\theta + \varphi) - g
    $$

    avec 

    $$
    \theta=0
    $$

    D'autre part, on a :

    $$
    y(t) = a t^3 + b t^2 + c t + d
    $$

    Avec les coefficients suivants :

    $$
    a = \frac{8}{125}, \quad b = -\frac{7}{25}, \quad c = -2, \quad d = 10
    $$

    En dérivant deux fois, on obtient :

    $$
    \ddot{y}(t) = 6a t + 2b = \frac{48}{125} t - \frac{14}{25}
    $$

    En remplaçant dans l'équation précédente, on obtient :

    $$
    \frac{f}{M} \cdot \cos(\varphi) = \frac{48}{125} t - \frac{14}{25} + g
    $$

    Donc :

    $$
    f = \left( \frac{48}{125} t - \frac{14}{25} + g \right) \cdot \frac{M}{\cos(\varphi)}
    $$

    Et en posant \( M = 1 \) kg et \( g = 1 \, \text{m/s}^2 \), on a :

    $$
    f = \left( \frac{48}{125} t + \frac{11}{25} \right) \cdot \frac{1}{\cos( \varphi)}
    $$

    """
    )
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


@app.cell
def _(np):
    def draw_flame(ax, x, y, theta, f, phi, M=1.0, g=1.0, l=1.0):
        flame_length = l * f / (M * g)
        flame_angle = theta + phi + np.pi
        x_end = x + flame_length * np.cos(flame_angle)
        y_end = y + flame_length * np.sin(flame_angle)
        ax.plot([x, x_end], [y, y_end], color="orange", linewidth=3)

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Visualization

    Produce a video of the booster for 5 seconds when

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=0$ and $\phi=0$

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=Mg$ and $\phi=0$

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=Mg$ and $\phi=\pi/8$

      - the parameters are those of the controlled landing studied above.

    As an intermediary step, you can begin with production of image snapshots of the booster location (every 1 sec).
    """
    )
    return


if __name__ == "__main__":
    app.run()
