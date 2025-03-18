import functools
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Protocol

import pygit2
from pygit2 import Blob, Commit, CredentialType, Oid, Reference, Repository, Tree

if TYPE_CHECKING:
    from pygit2 import GIT_OBJ_BLOB, GIT_OBJ_COMMIT
else:
    GIT_OBJ_COMMIT = pygit2.GIT_OBJECT_COMMIT
    GIT_OBJ_BLOB = pygit2.GIT_OBJECT_BLOB

parts = {
    "parking_controller": ("git@github.com:rss2025-4/parking_controller.git", "HEAD"),
}


def credentials_cb(url: str, username_from_url: str, allowed_types: CredentialType):
    return pygit2.KeypairFromAgent(username_from_url)


def main():
    remotecallback = pygit2.RemoteCallbacks(credentials=credentials_cb)
    repo = pygit2.init_repository(Path(__file__).parent / ".git")

    out_ref = "refs/heads/updated"

    this_script = repo.references.get("HEAD")
    assert this_script is not None
    this_script = this_script.peel(GIT_OBJ_COMMIT)

    final_tree_builder = repo.TreeBuilder(this_script.tree)

    def process_one_repo(name: str, url: str, ref: str) -> Oid:
        print("process_one_repo", name, url, ref)

        remote_name = f"_script_{name}"
        try:
            remote = repo.remotes[remote_name]
        except KeyError:
            remote = repo.remotes.create(name=remote_name, url=url)

        for ref_ in remote.ls_remotes(callbacks=remotecallback):
            print(ref_)

        local_ref = f"refs/remotes/{remote_name}/{ref}"
        remote.fetch(
            refspecs=[f"{ref}:{local_ref}"],
            callbacks=remotecallback,
        )
        commit_ = repo.references.get(local_ref)
        assert commit_ is not None
        commit = commit_.peel(GIT_OBJ_COMMIT)

        final_tree_builder.insert(name, commit.tree.id, pygit2.enums.FileMode.TREE)

        @functools.cache
        def map_commit(cur: Commit) -> Oid:
            print("map_commit", cur.short_id, cur.message.splitlines()[0])
            tree_builder = repo.TreeBuilder()
            tree_builder.insert(name, cur.tree.id, pygit2.enums.FileMode.TREE)
            return repo.create_commit(
                None,
                cur.author,
                cur.committer,
                cur.message,
                tree_builder.write(),
                [map_commit(x) for x in cur.parents],
            )

        return map_commit(commit)

    ans = [process_one_repo(name, remote, ref) for name, (remote, ref) in parts.items()]

    out_commit = repo.create_commit(
        None,
        this_script.author,
        this_script.committer,
        "update repositories",
        final_tree_builder.write(),
        [
            this_script.id,
            *[x for x in ans if not repo.descendant_of(this_script.id, x)],
        ],
    )
    repo.create_reference(out_ref, out_commit, force=True)


if __name__ == "__main__":
    main()
