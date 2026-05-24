from typing import Optional, TYPE_CHECKING
from world.entities.object import Object
from lib.types.netlogo_coordinate import NetLogoCoordinate

if TYPE_CHECKING:
    from world.managers.storage_manager import StorageManager
    from world.entities.pod import Pod

class Storage(Object):
    def __init__(self, id: int, x: int, y: int):
        super().__init__(id, 'storage', x, y)
        self.storage_manager: Optional[StorageManager] = None
        self.storage_number: int = id
        self.shape: str = 'full square'
        self.assigned_pod: Optional[Pod] = None
        self.is_empty: bool = True  # Default to empty

    def __eq__(self, other):
        if isinstance(other, Storage):
            return self.storage_number == other.storage_number
        return False

    def __hash__(self):
        return hash(self.storage_number)

    def __repr__(self):
        return f"Storage({self.storage_number})"

    def setStorageManager(self, storage_manager: "StorageManager"):
        self.storage_manager = storage_manager


    def setStoragePod(self, pod: "Pod"):
        self.assigned_pod = pod
        self.is_empty = False
        return

    def removeStoragePod(self):
        self.assigned_pod = None
        self.is_empty = True
        return

    def getPodInStorage(self) -> Optional["Pod"]:
        return self.assigned_pod