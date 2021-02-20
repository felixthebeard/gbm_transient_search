import subprocess

import luigi.format
from luigi.contrib.ssh import RemoteCalledProcessError
from luigi.contrib.ssh import RemoteContext as LuigiRemoteContext
from luigi.contrib.ssh import RemoteFileSystem as LuigiRemoteFileSystem
from luigi.contrib.ssh import RemoteTarget as LuigiRemoteTarget
from luigi.target import FileSystemTarget


class RemoteContext(LuigiRemoteContext):
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
