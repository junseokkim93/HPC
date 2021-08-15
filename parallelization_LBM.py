"""Performs a parallelization on application of LBM."""

# Standard library
import math
import time
import sys
from typing import List, Tuple

# Third party
import numpy as np
from mpi4py import MPI
from absl import flags
from absl import app

# defining flags
FLAGS = flags.FLAGS
flags.DEFINE_integer("Nx", None, "Integer value Nx from Ny x Nx process grid")
flags.DEFINE_integer("Ny", None, "Integer value Ny from Ny x Nx process grid")
flags.DEFINE_integer("Ndx", None, "Integer value Ndx from Ndy x Ndx decomposition grid")
flags.DEFINE_integer("Ndy", None, "Integer value Ndy from Ndy x Ndx decomposition grid")
# omega's default value is set to 1.7
flags.DEFINE_float("omega", 1.7, "The value for omega")
# t's default value is set to 100000
flags.DEFINE_integer("t", 100000, "Integer value for the timestep")
# required flags
flags.mark_flag_as_required("Nx")
flags.mark_flag_as_required("Ny")
flags.mark_flag_as_required("Ndx")
flags.mark_flag_as_required("Ndy")


# velocity sets
c = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1], [0, 0, 1, 0, -1, 1, 1, -1, -1]]).T
c_s = 1 / math.sqrt(3)

# number of channel
NC = 9

# weights for each channel
weights = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])

# wall velocity
u_w = 0.1




def main(argv) -> None:

    del argv

    # relaxation parameter
    omega = FLAGS.omega
    # number of steps
    t = FLAGS.t
    # grid size
    Nx = FLAGS.Nx
    Ny = FLAGS.Ny

    # decomposition
    Ndx = FLAGS.Ndx
    Ndy = FLAGS.Ndy

    # list of string containing wall information
    bounce_back_list = [
        "moving",
        "fixed",
        "fixed",
        "fixed",
    ]  # ("top", "right", "bottom", "left")

    # Functions for computation
    def equilibrium(rho: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """Takes density and velocity as inputs and return an initial equilibrium distribution.

        Args:
            rho (np.ndarray):  density of the particle, shape of (Nx,Ny).
            vel (np.ndarray):  average velocity, shape of (2,Nx,Ny).

        Returns:
            np.ndarray: initial equilibrium distribution.
        """

        # create a buffer for return value
        eqdf = np.zeros([NC, rho.shape[0], rho.shape[1]])
        u2 = np.einsum("ijk,ijk->jk", vel, vel)  # u*u

        for i, weight in zip(range(NC), weights):

            a = np.einsum("i,ijk->jk", c[i], vel)  # c_i*u
            a2 = np.multiply(a, a)  # a*a

            eqdf_part = weight * rho * (1 + 3 * a + 9 / 2 * a2 - (3 / 2) * u2)
            eqdf[i, :, :] = eqdf_part
            del a, a2, eqdf_part
        return eqdf

    def rho_calculate(f: np.ndarray) -> np.ndarray:
        """updates value of the density of the particle and return it

        Args:
            f (np.ndarray): probability distribution function, shape of (NC,Nx,Ny).

        Returns:
            np.ndarray: density of the particle, shape of (Nx,Ny).
        """
        return np.einsum("ijk->jk", f)

    def vel_calculate(f: np.ndarray, rho: np.ndarray) -> np.ndarray:
        """updates value of the average velocity and return it

        Args:
            f (np.ndarray): probability distribution function, shape of (NC,Nx,Ny).
            rho (np.ndarray): density of the particle, shape of (Nx,Ny).

        Returns:
            np.ndarray: average velocity, shape of (2,Nx,Ny).
        """
        return np.einsum("ji,jkl->ikl", c, f) / rho

    def streaming(f: np.ndarray) -> None:
        """performs a streaming operation.

        Args:
            f (np.ndarray): probability distribution function, shape of (NC,Nx,Ny).
        """
        for i in range(1, 9):
            f[i] = np.roll(f[i], c[i], axis=(0, 1))

    def collision(f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """performs a collision operation.

        Args:
            f (np.ndarray): probability distribution function, shape of (NC,Nx,Ny).

        Returns:
            Tuple[np.ndarray, np.ndarray]: tuple consisting of density and velocity matrices.
        """
        rho = rho_calculate(f)
        vel = vel_calculate(f, rho)
        f += (omega) * (equilibrium(rho, vel) - f)
        return rho, vel

    def stream_and_bounce_back(
        f: np.ndarray,
        bounce_back_list: List[str] = [None, None, None, None],
        u_w: float = None,
        corner: bool = False,
    ) -> None:
        """Given the hyperparameter including speed of a wall, and bounce_back_list which includes
        the string for each position with either "moving" or "fixed", this function sets bounce back
        boundary conditon and also operates streaming function which is also required.

        Args:
            f (np.ndarray): probability distribution function, shape of (NC,Nx,Ny).

            bounce_back_list (List[str], optional): list of strings, Each position represents
                "top", "right", "bottom", "left" respectively. Either "moving" or "fixed" can be input.
                Defaults to [None, None, None, None].

            u_w (float, optional): speed of the wall. Defaults to None.

            corner (bool, optional): Bool variable which decides
                whether the corner shall be considered or not. Defaults to False.

        Raises:
            Exception: when the wall moves but no wall speed is given, the error occurs.
        """
        if "move" in bounce_back_list and u_w == None:
            raise Exception("u_w cannot be None when moving wall is given.")

        # copy edge part of distribution function before streaming
        f_bottom = f[:, :, 0].copy()
        f_top = f[:, :, -1].copy()
        f_left = f[:, 0, :].copy()
        f_right = f[:, -1, :].copy()

        # streaming operation
        streaming(f)

        # to keep track of making sure our bounce_back case exists
        match = 0

        # Bottom: fixed wall
        if not bounce_back_list[2] == None and bounce_back_list[2].lower() == "fixed":
            # "fixed bottom"
            f[[2, 5, 6], :, 0] = f_bottom[[4, 7, 8]]
            match += 1

        # Top: fixed wall
        if not bounce_back_list[0] == None and bounce_back_list[0].lower() == "fixed":
            # "fixed top"
            f[[4, 7, 8], :, -1] = f_top[[2, 5, 6]]
            match += 1

        # Top: moving wall
        elif (
            not bounce_back_list[0] == None and bounce_back_list[0].lower() == "moving"
        ):
            # "moving top"
            # rho_wall's 4,7,8 channel comes from the former f_top's 2,5,6
            rho_wall = f_top[[2, 5, 6]] + f[[2, 5, 6], :, -1] + f[[0, 1, 3], :, -1]
            rho_wall = np.einsum("ij->j", rho_wall)
            f[4, :, -1] = f_top[2]
            f[8, :, -1] = f_top[6] + 6 * weights[6] * rho_wall * u_w
            f[7, :, -1] = f_top[5] - 6 * weights[5] * rho_wall * u_w
            match += 1

        # Left: fixed wall
        if not bounce_back_list[3] == None and bounce_back_list[3].lower() == "fixed":
            # "fixed left"
            f[[1, 5, 8], 0, :] = f_left[[3, 7, 6]]
            match += 1

        # Right: fixed wall
        if not bounce_back_list[1] == None and bounce_back_list[1].lower() == "fixed":
            # "fixed right"
            f[[3, 7, 6], -1, :] = f_right[[1, 5, 8]]
            match += 1

        if corner:

            # Bottom: corner_left
            f[[2, 1, 5], 0, 0] = f_bottom[[4, 3, 7], 0]
            # Bottom: corner_right
            f[[2, 3, 6], -1, 0] = f_bottom[[4, 1, 8], -1]

            # Top: corner_left
            f[[4, 1, 8], 0, -1] = f_top[[2, 3, 6], 0]
            # Top: corner_right
            f[[4, 3, 7], -1, -1] = f_top[[2, 1, 5], -1]

    def communicate(f):
        """
        Communicate boundary regions to ghost regions.

        Parameters
        ----------
        f : array
            Array containing the occupation numbers. Array is 3-dimensional, with
            the first dimension running from 0 to 8 and indicating channel. The
            next two dimensions are x- and y-position. This array is modified in
            place.
        """
        # Send to left
        recvbuf = f[:, -1, :].copy()
        comm.Sendrecv(f[:, 1, :].copy(), left_dst, recvbuf=recvbuf, source=left_src)
        f[:, -1, :] = recvbuf
        # Send to right
        recvbuf = f[:, 0, :].copy()
        comm.Sendrecv(f[:, -2, :].copy(), right_dst, recvbuf=recvbuf, source=right_src)
        f[:, 0, :] = recvbuf
        # Send to bottom
        recvbuf = f[:, :, -1].copy()
        comm.Sendrecv(f[:, :, 1].copy(), bottom_dst, recvbuf=recvbuf, source=bottom_src)
        f[:, :, -1] = recvbuf
        # Send to top
        recvbuf = f[:, :, 0].copy()
        comm.Sendrecv(f[:, :, -2].copy(), top_dst, recvbuf=recvbuf, source=top_src)
        f[:, :, 0] = recvbuf

    def save_mpiio(comm, fn, g_kl):
        """
        Write a global two-dimensional array to a single file in the npy format
        using MPI I/O: https://docs.scipy.org/doc/numpy/neps/npy-format.html

        Arrays written with this function can be read with numpy.load.

        Parameters
        ----------
        comm
            MPI communicator.
        fn : str
            File name.
        g_kl : array
            Portion of the array on this MPI processes.
        """
        from numpy.lib.format import dtype_to_descr, magic

        magic_str = magic(1, 0)

        local_nx, local_ny = g_kl.shape
        nx = np.empty_like(local_nx)
        ny = np.empty_like(local_ny)

        commx = comm.Sub((True, False))
        commy = comm.Sub((False, True))
        commx.Allreduce(np.asarray(local_nx), nx)
        commy.Allreduce(np.asarray(local_ny), ny)

        arr_dict_str = str(
            {
                "descr": dtype_to_descr(g_kl.dtype),
                "fortran_order": False,
                "shape": (np.asscalar(nx), np.asscalar(ny)),
            }
        )
        while (len(arr_dict_str) + len(magic_str) + 2) % 16 != 15:
            arr_dict_str += " "
        arr_dict_str += "\n"
        header_len = len(arr_dict_str) + len(magic_str) + 2

        offsetx = np.zeros_like(local_nx)
        commx.Exscan(np.asarray(ny * local_nx), offsetx)
        offsety = np.zeros_like(local_ny)
        commy.Exscan(np.asarray(local_ny), offsety)

        file = MPI.File.Open(comm, fn, MPI.MODE_CREATE | MPI.MODE_WRONLY)
        if MPI.COMM_WORLD.Get_rank() == 0:
            file.Write(magic_str)
            file.Write(np.int16(len(arr_dict_str)))
            file.Write(arr_dict_str.encode("latin-1"))
        mpitype = MPI._typedict[g_kl.dtype.char]
        filetype = mpitype.Create_vector(g_kl.shape[0], g_kl.shape[1], ny)
        filetype.Commit()
        file.Set_view(
            header_len + (offsety + offsetx) * mpitype.Get_size(), filetype=filetype
        )
        file.Write_all(g_kl.copy())
        filetype.Free()
        file.Close()

    ### Initialize MPI communicator
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        print("Running in parallel on {} MPI processes.".format(size))
    assert Ndx * Ndy == size
    if rank == 0:
        print("Domain decomposition: {} x {} MPI processes.".format(Ndx, Ndy))
        print("Global grid has size {}x{}.".format(Nx, Ny))

    # Create cartesian communicator and get MPI ranks of neighboring cells
    comm = MPI.COMM_WORLD.Create_cart((Ndx, Ndy), periods=(False, False))
    left_src, left_dst = comm.Shift(0, -1)
    right_src, right_dst = comm.Shift(0, 1)
    bottom_src, bottom_dst = comm.Shift(1, -1)
    top_src, top_dst = comm.Shift(1, 1)

    local_Nx = Nx // Ndx
    local_Ny = Ny // Ndy
    # We need to take care that the total number of *local* grid points sums up to
    # Nx. The right and topmost MPI processes are adjusted such that this is
    # fulfilled even if Nx, Ny is not divisible by the number of MPI processes.
    if right_dst < 0:
        # This is the rightmost MPI process
        local_Nx = Nx - local_Nx * (Ndx - 1)
    without_ghosts_x = slice(0, local_Nx)
    if right_dst >= 0:
        # Add ghost cell
        local_Nx += 1
    if left_dst >= 0:
        # Add ghost cell
        local_Nx += 1
        without_ghosts_x = slice(1, local_Nx + 1)
    if top_dst < 0:
        # This is the topmost MPI process
        local_Ny = Ny - local_Ny * (Ndy - 1)
    without_ghosts_y = slice(0, local_Ny)
    if top_dst >= 0:
        # Add ghost cell
        local_Ny += 1
    if bottom_dst >= 0:
        # Add ghost cell
        local_Ny += 1
        without_ghosts_y = slice(1, local_Ny + 1)
    mpix, mpiy = comm.Get_coords(rank)
    print(
        "Rank {} has domain coordinates {}x{} and a local grid of size {}x{} (including ghost cells).".format(
            rank, mpix, mpiy, local_Nx, local_Ny
        )
    )

    f = equilibrium(
        np.ones((local_Nx, local_Ny), dtype=np.float64),
        np.zeros((2, local_Nx, local_Ny), dtype=np.float64),
    )

    # main loop
    start = time.time()
    for del_t in range(t):
        if del_t % 10 == 9:
            sys.stdout.write("=== Step {}/{} ===\r".format(del_t + 1, t))
        communicate(f)

        stream_and_bounce_back(
            f, bounce_back_list=bounce_back_list, u_w=u_w, corner=True
        )
        rho, vel = collision(f)
    if mpix==0:
        save_mpiio(
            comm, "ux_{}X{}_{}.npy".format(Nx,Ny,del_t), vel[0, 0:-1, 1:-1]
        )
        save_mpiio(
            comm, "uy_{}X{}_{}.npy".format(Nx,Ny,del_t), vel[1, 0:-1, 1:-1]
        )
#     elif mpix==-1:
#         save_mpiio(
#             comm, "ux_{}X{}_{}.npy".format(Nx,Ny,del_t), vel[0, 1:-1, 1:-1]
#         )
#         save_mpiio(
#             comm, "uy_{}X{}_{}.npy".format(Nx,Ny,del_t), vel[1, 1:-1, 1:-1]
#         )
    elif mpiy==0:
        save_mpiio(
            comm, "ux_{}X{}_{}.npy".format(Nx,Ny,del_t), vel[0, 1:-1, 0:-1]
        )
        save_mpiio(
            comm, "uy_{}X{}_{}.npy".format(Nx,Ny,del_t), vel[1, 1:-1, 0:-1]
        )
#     elif mpix==0:
#         save_mpiio(
#             comm, "ux_{}X{}_{}.npy".format(Nx,Ny,del_t), vel[0, 1:-1, 1:-1]
#         )
#         save_mpiio(
#             comm, "uy_{}X{}_{}.npy".format(Nx,Ny,del_t), vel[1, 1:-1, 1:-1]
#         )
    # measure how long it took to run
    time_took = time.time() - start
    # calculate MLUPS(million lattice update per second)
    MLUPS = t * Nx * Ny / time_took / 1e6
    
    if rank==0: print("\nWith {} MPI processors,{}X{} grid, MLUPS: {}\n".format(size,Nx,Ny, MLUPS))
    
    
    

if __name__ == "__main__":
    app.run(main)
