"""
Copyright 2023, University of Freiburg
Chair of Algorithms and Data Structures.
Hannah Bast <bast@cs.uni-freiburg.de>
Patrick Brosi <brosi@cs.uni-freiburg.de>
Natalie Prange <prange@cs.uni-freiburg.de>
Sebastian Walter <swalter@cs.uni-freiburg.de>
"""

import argparse
import readline  # noqa
import socket
import time
import mimetypes
from pathlib import Path

from sparql_to_sql import SPARQL
try:
    # try to import the ad_freiburg_qgram_utils package,
    # which contains a faster Rust-based implementation of a q-gram index;
    # install it via pip install ad-freiburg-qgram-utils
    from ad_freiburg_qgram_utils import QGramIndex  # type: ignore
except ImportError:
    # fallback to the Python implementation in qgram_index.py
    # if ad_freiburg_qgram_utils is not installed
    from qgram_index import QGramIndex  # type: ignore


class Server:
    """

    A HTTP server using a q-gram index and SPARQL engine.

    In the following, you will find some example URLs for the search
    and relations APIs, each given with the expected JSON output.
    Note that, as usual, the contents of the test cases is important,
    but not the exact syntax.

    URL:
      http://<host>:<port>/api/search?q=angel
    RESPONSE:
      {
        "numTotalResults": 2152,
        "results": [
          {
            "name": "Angela Merkel",
            "score": 211,
            "description": "chancellor of Germany from 2005 to 2021"
          },
          {
            "name": "Angelina Jolie",
            "score": 160,
            "description": "American actress (born 1975)"
          },
          {
            "name": "angel",
            "score": 147,
            "description": "supernatural being or spirit in certain religions \
            and mythologies"
          },
          {
            "name": "Angel Falls",
            "score": 91,
            "description": "waterfall in Venezuela; \
            highest uninterrupted waterfall in the world"
          },
          {
            "name": "Angela Davis",
            "score": 73,
            "description": "American political activist, scholar, and author"
          }
        ]
      }

    URL:
      http://<host>:<port>/api/search?q=eyj%C3%A4fja
    RESPONSE:
      {
        "numTotalResults": 4,
        "results": [
          {
            "name": "Eyjafjallajökull",
            "score": 77,
            "description": "ice cap in Iceland covering the caldera of a \
            volcano"
          },
          {
            "name": "Eyjafjallajökull",
            "score": 24,
            "description": "volcano in Iceland"
          },
          {
            "name": "Eyjafjarðarsveit",
            "score": 21,
            "description": "municipality of Iceland"
          },
          {
            "name": "Eyjafjallajökull",
            "score": 8,
            "description": "2013 film by Alexandre Coffre"
          }
        ]
      }

    URL:
      http://<host>:<port>/api/relations?id=Q567
    RESPONSE:
      [
        {
            "predicate" : "instance of",
            "object": "human"
        },
        {
            "predicate" : "occupation",
            "object": "physicist, politician"
        },
        {
            "predicate" : "sex or gender",
            "object": "female"
        },
        {
            "predicate" : "given name",
            "object": "Angela"
        },
        {
            "predicate" : "country of citizenship",
            "object": "Germany"
        },
        {
            "predicate" : "place of birth",
            "object": "Eimsbüttel"
        },
        {
            "predicate" : "languages spoken, written or signed",
            "object": "German"
        },
        {
            "predicate" : "educated at",
            "object": "Leipzig University, Academy of Sciences of the GDR"
        },
        {
            "predicate" : "award received",
            "object": "Jawaharlal Nehru Award for International Understanding,\
            Order of Stara Planina, Robert Schuman Medal, Order of the \
            Republic, Order of Zayed, Bavarian Order of Merit, Order of Merit \
            of the Italian Republic, Order of King Abdulaziz al Saud, Order \
            of Vytautas the Great, Félix Houphouët-Boigny Peace Prize, \
            Order of Liberty, Order of the Three Stars, Presidential Medal of \
            Distinction, Supreme Order of the Renaissance, Time Person of the \
            Year, Nansen Refugee Award, Financial Times Person of the Year, \
            Charlemagne Prize, Presidential Medal of Freedom"
        },
        {
            "predicate" : "position held",
            "object": "Federal Chancellor of Germany"
        },
        {
            "predicate" : "father",
            "object": "Horst Kasner"
        },
        {
            "predicate" : "field of work",
            "object": "analytical chemistry, theoretical chemistry, \
            politics, physics"
        },
        {
            "predicate" : "participant in",
            "object": "2012 German presidential election, 2009 German \
            presidential election, 2017 German presidential election, \
            2010 German presidential election"
        },
        {
            "predicate" : "spouse",
            "object": "Joachim Sauer"
        },
        {
            "predicate" : "topic's main category",
            "object": "Category:Angela Merkel"
        },
        {
            "predicate" : "member of",
            "object": "Fourth Merkel cabinet, Third Merkel cabinet, Cabinet \
            Kohl IV, Cabinet Kohl V, Free German Youth, First Merkel cabinet, \
            Second Merkel cabinet"
        }
      ]

    URL:
      http://<host>:<port>/api/relations?id=Q39651
    RESPONSE:
      [
        {
            "predicate" : "instance of",
            "object": "ice cap"
        },
        {
            "predicate" : "country",
            "object": "Iceland"
        },
        {
            "predicate" : "located in the administrative territorial entity",
            "object": "Rangárþing eystra, Southern Region"
        },
        {
            "predicate" : "located in time zone",
            "object": "UTC±00:00"
        },
        {
            "predicate" : "different from",
            "object": "Eyjafjallajökull"
        },
        {
            "predicate" : "topic's main category",
            "object": "Category:Eyjafjallajökull"
        }
      ]
    """

    def __init__(
        self,
        port: int,
        qi: QGramIndex,
        db: str,
        party_pooper: bool = False
    ) -> None:
        """

        Initializes a simple HTTP server with
        the given q-gram index, database and port.

        """
        self.port = port
        self.qi = qi
        self.db = db
        self.engine = SPARQL()
        self.party_pooper = party_pooper

    def run(self) -> None:
        """

        Runs the server loop:
        Creates a socket, and then, in an infinite loop,
        waits for requests and processes them.

        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", self.port))
            sock.listen(1)

            while True:
                print(f"Waiting on port {self.port}")
                conn, client_address = sock.accept()
                # connection timeout 5 seconds
                conn.settimeout(5.0)
                try:
                    print(f"Client connected from {client_address[0]}")

                    data = bytearray()
                    end = -1
                    while end < 0:
                        batch = conn.recv(32)

                        if batch:
                            data.extend(batch)
                            end = data.find(b'\r\n\r\n')

                    self.handle_request(conn, data[:end])
                except socket.timeout:
                    print(f"(Timeout from {client_address[0]})")
                    pass
                except Exception as e:
                    print(f"Unexpected exception: {e}")
                    raise e
                finally:
                    conn.close()
        finally:
            print("Closing socket...")
            sock.close()

    def handle_request(
        self,
        con: socket.socket,
        req: bytearray
    ) -> None:
        """

        Handles the given http requests and sends the result via the given
        connection.

        """
        print(f"Handling request: {req.decode('utf8')}")

        meth, path, *_ = req.decode("utf8").split(" ")
        # we only support GET
        if meth != "GET":
            con.sendall(
                b"HTTP/1.1 405 Not Allowed\r\n"
                b"Content-length: 12\r\n\r\n"
                b"Not allowed!"
            )
            return

        split = path.split("?")
        path = split[0]

        params = {}
        if len(split) > 1:
            for pair in split[1].split("&"):
                key, value = pair.split("=")
                params[key] = value

        # if API request is received
        if path == "/api/search":
            self.handle_search_api_request(con, params)
        elif path == "/api/relations":
            self.handle_relations_api_request(con, params)

        # ensure path is relative to CWD/resources/
        filep = Path(__file__).parent.absolute().joinpath(
            Path("./resources/" + path)
        )

        # return 404 if file does not exist
        if not filep.exists():
            con.sendall(
                b"HTTP/1.1 404 Not found\r\n"
                b"Content-length: 10\r\n\r\n"
                b"Not found!"
            )
            return

        # return 403 for dir requests
        if not filep.is_file():
            con.sendall(
                b"HTTP/1.1 403 Forbidden\r\n"
                b"Content-length: 12\r\n\r\n"
                b"Not allowed!"
            )
            return

        # guess MIME type
        mimetype, _ = mimetypes.guess_type(
            str(filep.resolve())
        ) or "text/plain"

        # read request file as bytearray
        with filep.open("rb") as reqf:
            byte_content = reqf.read()

        if filep.name == "search.html":
            # handle searches
            content = byte_content.decode("utf8")

            # back to bytearray
            byte_content = bytearray(content, "utf8")

        # send header and content
        header = bytearray(
            f"HTTP/1.1 200 OK\r\n"
            f"Content-type: {mimetype}\r\n"
            f"Content-length: {len(byte_content)}\r\n\r\n",
            "utf8"
        )
        con.sendall(header + byte_content)

    def handle_search_api_request(self, con, params):
        # send header and content
        response = self.get_search_results(params).encode("utf-8")
        header = bytearray(
            f"HTTP/1.1 200 OK\r\n"
            f"Content-type: application/json\r\n"
            f"Content-length: {len(response)}\r\n\r\n",
            "utf8"
        )
        con.sendall(header + response)

    def handle_relations_api_request(self, con, params):
        # send header and content
        response = self.get_relations_results(params).encode("utf-8")
        header = bytearray(
            f"HTTP/1.1 200 OK\r\n"
            f"Content-type: application/json\r\n"
            f"Content-length: {len(response)}\r\n\r\n",
            "utf8"
        )
        con.sendall(header + response)

    def get_relations_results(self, params):
        wikidata_id = params.get("id")
        sql = self.engine.sparql_to_sql(
            f"""
            SELECT ?p_name ?o_name WHERE {{
                wdt:{wikidata_id} ?p ?o .
                ?p rdfs:label ?p_name .
                ?o rdfs:label ?o_name .
                ?p custom:count ?p_count
            }} ORDER BY DESC(?p_count)
            """
        )
        grouped = {}
        for p, o in self.engine.process_sql_query(
            self.db,
            sql
        ):
            if p not in grouped:
                grouped[p] = {o}
            else:
                grouped[p].add(o)
        print(grouped)
        json = f"""{{
                {[
            (pred, ", ".join(values))
            for pred, values in grouped.items()]}
            }}"""

        return json

    def get_search_results(self, params):
      # get query result from dataset
      q = params.get("q", "")
      query = self.qi.normalize(q)

      if query != "":
          delta = len(query) // (self.qi.q + 1)
          postings = self.qi.find_matches(query, delta)
      else:
          postings = []

      # prepare entities
      entities = []
      for i, (id, ped) in enumerate(postings[:5]):
          infos = self.qi.get_infos(id)
          if infos is None:
              continue

          syn, name, score, info = infos
          wikidata_id, description, wikipedia_url, img_url = info

          entities.append({
            "entity_id": str(i),
            "entity_name": name,
            "entity_synonym": syn,
            "entity_score": str(score),
            "entity_ped": str(ped),
            "entity_desc": description,
            "entity_img": img_url or "noimage.png",
            "wikidata_url": f"https://www.wikidata.org/wiki/{wikidata_id}",
            "wikipedia_url": wikipedia_url
          })

      json = f"""{{
                "results": {len(postings)},
                "entities": {entities}
            }}"""

      return json

    def url_decode(self, string: str) -> str:
        """

        Decodes an URL-encoded UTF-8 string, as explained in the lecture.
        Also decodes "+" to " " (space).

        >>> s = Server(0, None, "")
        >>> s.url_decode("nirwana")
        'nirwana'
        >>> s.url_decode("the+m%C3%A4trix")
        'the mätrix'
        >>> s.url_decode("Mikr%C3%B6soft+Windos")
        'Mikrösoft Windos'
        >>> s.url_decode("The+hitschheiker%20guide")
        'The hitschheiker guide'
        """

        byte_array = bytearray()
        i = 0
        while i < len(string):
          if string[i] == "+":
              byte_array.extend(b" ")
          elif string[i] == "%":
              hex = string[i+1:i+3]
              byte_array.extend(bytes.fromhex(hex))
              i += 2
          else:
              byte_array.extend(bytearray(string[i], "utf-8"))
          i += 1

        return byte_array.decode("utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "entities",
        type=str,
        help="path to entities file for q-gram index"
    )
    parser.add_argument(
        "db",
        type=str,
        help="path to sqlite3 database for SPARQL engine"
    )
    parser.add_argument(
        "port",
        type=int,
        help="port to run the server on"
    )
    parser.add_argument(
        "-q",
        "--q-grams",
        type=int,
        default=3,
        help="size of the q-grams"
    )
    parser.add_argument(
        "-s",
        "--use-synonyms",
        action="store_true",
        help="whether to use synonyms for the q-gram index"
    )
    parser.add_argument(
        "-p",
        "--party-pooper",
        action="store_true",
        help="whether to prevent code injection"
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    # Create a new q-gram index from the given file.
    print(f"Building q-gram index from file {args.entities}.")
    start = time.perf_counter()
    q = QGramIndex(args.q_grams, args.use_synonyms)
    q.build_from_file(args.entities)
    print(f"Done, took {(time.perf_counter() - start) * 1000:.1f}ms.")

    server = Server(
        args.port,
        q,
        args.db,
        args.party_pooper
    )
    print(
        f"Starting server on port {args.port}, go to "
        f"http://localhost:{args.port}/search.html"
    )
    server.run()


if __name__ == "__main__":
    main(parse_args())
