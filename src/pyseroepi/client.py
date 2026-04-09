"""
Module to interact with the Pathogenwatch Next API.
"""
from dataclasses import dataclass, field, fields
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Generator, Optional, ClassVar, Iterable
import concurrent.futures


# Classes --------------------------------------------------------------------------------------------------------------
class PathogenwatchClient:
    """
    Client for the Pathogenwatch Next API.
    Handles automatic retries, rate limiting, and pagination.
    """
    _BASE = "https://next.pathogen.watch/api/"
    _COLLECTIONS_ENDPOINT = "collections/list"
    _FOLDERS_ENDPOINT = "folders/list"

    def __init__(self, api_key: str):
        self.session = requests.Session()
        
        # Set authentication
        self.session.headers.update({
            "X-API-Key": api_key,
            "Content-Type": "application/json",
            "User-Agent": "pyseroepi-client/1.0"
        })
        
        # This replaces the entire threading/lock/backoff mechanism of the old template.
        # It automatically pauses and retries on rate limits (429) or server errors (50X).
        retries = Retry(
            total=5,
            backoff_factor=1,  # 1s, 2s, 4s, 8s, 16s
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=20)
        self.session.mount("https://", adapter)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.close()

    def prefetch(self, items: Iterable['PathogenwatchContainerMixin'], max_workers: int = 10) -> None:
        """
        Concurrently populates the details and genomes cache for multiple collections/folders.
        Uses thread pooling to fetch in parallel while urllib3 safely handles 429 rate limit backoffs.
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submitting item.get_genomes naturally triggers item.get_details 
            # resolving and caching both sequentially per-item, but concurrently across items.
            futures = [executor.submit(item.get_genomes, self) for item in items]
            for future in concurrent.futures.as_completed(futures):
                future.result()  # Raise any exceptions encountered during fetching

    def request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        url = f"{self._BASE}/{endpoint.lstrip('/')}"
        with self.session.request(method, url, **kwargs) as response:
            response.raise_for_status() # Automatically raises an error for 4xx/5xx responses
            return response

    def get(self, endpoint: str, **kwargs) -> requests.Response:
        url = f"{self._BASE}/{endpoint.lstrip('/')}"
        with self.session.get(url, **kwargs) as response:
            response.raise_for_status() # Automatically raises an error for 4xx/5xx responses
            return response

    def get_collections(self, exclude: str = None, limit: int = None, binned: bool = None
    ) -> Generator['PathogenwatchCollection', None, None]:
        params = {k: v for k, v in [("exclude", exclude), ("limit", limit),
                                    ("binned", str(binned).lower() if binned is not None else None)] if
                  v is not None}

        valid_keys = {f.name for f in fields(PathogenwatchCollection) if f.init}
        for collection_dict in self.get(self._COLLECTIONS_ENDPOINT, params=params).json():
            yield PathogenwatchCollection(**{k: v for k, v in collection_dict.items() if k in valid_keys})

    def get_folders(self, exclude: str = None, limit: int = None, binned: bool = None
                    ) -> Generator['PathogenwatchFolder', None, None]:
        params = {k: v for k, v in [("exclude", exclude), ("limit", limit),
                                    ("binned", str(binned).lower() if binned is not None else None)] if
                  v is not None}

        valid_keys = {f.name for f in fields(PathogenwatchFolder) if f.init}
        for folder_dict in self.get(self._FOLDERS_ENDPOINT, params=params).json():
            yield PathogenwatchFolder(**{k: v for k, v in folder_dict.items() if k in valid_keys})



class PathogenwatchContainerMixin:
    """
    Mixin providing shared fetching logic for Pathogenwatch Collection and Folder dataclasses.
    Classes using this mixin must define _ENTITY_TYPE, _DETAILS_QUERY_PARAM,
    _GENOMES_ID_PARAM, _GENOMES_CURSOR_PARAM, and _ATTR_PREFIX.
    """
    _ENTITY_TYPE: ClassVar[str]
    _DETAILS_QUERY_PARAM: ClassVar[str]
    _GENOMES_ID_PARAM: ClassVar[str]
    _GENOMES_CURSOR_PARAM: ClassVar[str]
    _ATTR_PREFIX: ClassVar[str]

    def get_details(self, client: PathogenwatchClient) -> dict:
     if self._details_cache is None:
         details = client.get(f"{self._ENTITY_TYPE}/details", params={self._DETAILS_QUERY_PARAM: self.uuid}).json()
         object.__setattr__(self, '_details_cache', details)
     return self._details_cache

    def get_genomes(self, client: PathogenwatchClient, limit: int = 1000) -> list[dict]:

     internal_id = self.get_details(client).get('id')
     if not internal_id:
         raise ValueError(f"Could not resolve internal ID for {self._ENTITY_TYPE[:-1]} {self.uuid}")

     all_genomes = []
     cursor = None

     while True:
         params = {self._GENOMES_ID_PARAM: internal_id, "limit": limit}
         if cursor:
             params[self._GENOMES_CURSOR_PARAM] = cursor

         data = client.get(f"{self._ENTITY_TYPE}/genomes", params=params).json()
         all_genomes.extend(data.get("genomes", []))

         cursor = data.get("meta", {}).get("endCursor")
         if not cursor or data.get("meta", {}).get("empty"):
             break

     return all_genomes


@dataclass(frozen=True, slots=True)
class PathogenwatchCollection(PathogenwatchContainerMixin):
    """
    A lazy-loaded proxy object representing a single Pathogenwatch collection.
    """
    _ENTITY_TYPE: ClassVar[str] = "collections"
    _DETAILS_QUERY_PARAM: ClassVar[str] = "uuid"
    _GENOMES_ID_PARAM: ClassVar[str] = "collectionId"
    _GENOMES_CURSOR_PARAM: ClassVar[str] = "cursor"
    _ATTR_PREFIX: ClassVar[str] = "pw_collection"
    
    binned: bool
    createdAt: str
    description: str
    name: str
    organismId: str
    owner: str
    uuid: str
    size: int
    _details_cache: Optional[dict] = field(default=None, init=False, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class PathogenwatchFolder(PathogenwatchContainerMixin):
    """
    A lazy-loaded proxy object representing a single Pathogenwatch folder.
    """
    _ENTITY_TYPE: ClassVar[str] = "folders"
    _DETAILS_QUERY_PARAM: ClassVar[str] = "id"
    _GENOMES_ID_PARAM: ClassVar[str] = "folderId"
    _GENOMES_CURSOR_PARAM: ClassVar[str] = "after"
    _ATTR_PREFIX: ClassVar[str] = "pw_folder"

    createdAt: str
    id: str
    uuid: str
    access: str
    name: str = ""
    binned: bool = False
    _details_cache: Optional[dict] = field(default=None, init=False, repr=False, compare=False)


# def test():
#     client = PathogenwatchClient('')
#     collections = list(client.get_collections())
#     collection = next((i for i in collections if 'sepsis' in i.name), None)
#     genomes = collection.get_genomes(client)
#
#     # dist = Distances.from_pathogenwatch()
#
#     # from pyseroepi import PathogenwatchParser
#     # df = PathogenwatchParser().from_records(genomes)
#     import pandas as pd
#     df = pd.DataFrame(genomes).set_index("id")
#
