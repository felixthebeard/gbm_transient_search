import subprocess
import os
import logging

import luigi.format
from luigi.contrib.ssh import RemoteCalledProcessError
from luigi.contrib.ssh import RemoteContext as LuigiRemoteContext
from luigi.contrib.ssh import RemoteFileSystem as LuigiRemoteFileSystem
from luigi.contrib.ssh import RemoteTarget as LuigiRemoteTarget
from luigi.target import FileSystemTarget
from gbm_bkg_pipe.configuration import gbm_bkg_pipe_config

socket_base_path = gbm_bkg_pipe_config["ssh"].get("master_socket_base_path", None)
nr_sockets = gbm_bkg_pipe_config["ssh"].get("nr_sockets", 1)


class RemoteContext(LuigiRemoteContext):
    @property
    def master_socket_paths(self):
        sockets = []

        for i in range(nr_sockets):
            socket_path = os.path.join(socket_base_path, f"{self._host_ref()}_{i}:22")

            if os.path.exists(socket_path):
                sockets.append(socket_path)

            else:
                logging.error(
                    f"The master socket path is not existing at {socket_path}."
                    f"It has the be created manually."
                )
        return sockets

    def check_nr_of_channels(self, master_socket):
        output = subprocess.check_output(
            f"lsof -U | grep {master_socket} | wc -l", shell=True
        ).decode()
        nr_channels = int(output.strip(" \n"))
        return nr_channels

    def get_free_socket(self):
        sockets = []

        for socket in self.master_socket_paths:
            sockets.append((socket, self.check_nr_of_channels(socket)))

        sockets.sort(key=lambda tup: tup[1])
        most_free_socket = sockets[0]

        if most_free_socket[1] > 8:
            raise Exception(
                "The master socket with the least number of connections has more than 8!"
            )

        return ["-S", most_free_socket[0]]

    def _prepare_cmd(self, cmd):
        connection_cmd = ["ssh", self._host_ref()]

        if socket_base_path is None:
            connection_cmd += ["-o", "ControlMaster=no"]
        else:
            connection_cmd += self.get_free_socket()

        if self.sshpass:
            connection_cmd = ["sshpass", "-e"] + connection_cmd
        else:
            connection_cmd += ["-o", "BatchMode=yes"]  # no password prompts etc
        if self.port:
            connection_cmd.extend(["-p", self.port])

        if self.connect_timeout is not None:
            connection_cmd += ["-o", "ConnectTimeout=%d" % self.connect_timeout]

        if self.no_host_key_check:
            connection_cmd += [
                "-o",
                "UserKnownHostsFile=/dev/null",
                "-o",
                "StrictHostKeyChecking=no",
            ]

        if self.key_file:
            connection_cmd.extend(["-i", self.key_file])

        if self.tty:
            connection_cmd.append("-t")
        return connection_cmd + cmd

    def check_output(self, cmd):
        p = self.Popen(cmd, stdout=subprocess.PIPE)

        output, _ = p.communicate()

        if p.returncode != 0:
            raise RemoteCalledProcessError(p.returncode, cmd, self.host, output=output)

        try:
            p.terminate()
        except Exception as e:
            print(e)

        _ = super().check_output(["exit"])

        return output


class RemoteFileSystem(LuigiRemoteFileSystem):
    def __init__(self, host, **kwargs):
        self.remote_context = RemoteContext(host, **kwargs)


class RemoteTarget(LuigiRemoteTarget):
    """
    Target used for reading from remote files.

    The target is implemented using ssh commands streaming data over the network.
    """

    def __init__(self, path, host, format=None, **kwargs):
        FileSystemTarget.__init__(self, path)
        if format is None:
            format = luigi.format.get_default_format()
        self.format = format
        self._fs = RemoteFileSystem(host, **kwargs)
