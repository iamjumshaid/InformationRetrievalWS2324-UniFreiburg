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
from pathlib import Path
import mimetypes

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

    A HTTP server using a q-gram index and SPARQL engine (optional).

    No pre-defined tests are required this time. However, if you add new
    non-trivial methods, you should of course write tests for them.

    Your server should behave like explained in the lecture. For a given
    URL of the form http://<host>:<port>/search.html?q=<query>, your server
    should return a (static) HTML page that displays (1) an input field and a
    search button as shown in the lecture, (2) the query without any URL
    encoding characters and (3) the top-5 entities returned by a q-gram
    index for the query.

    In the following, you will find some example URLs, each given with the
    expected query (%QUERY%) and the expected entities (%RESULT%, each in the
    format "<name>;<score>;<description>") that should be displayed by the
    HTML page returned by your server when calling the URL. Note that, as
    usual, the contents of the test cases is important, but not the exact
    syntax. In particular, there is no HTML markup given, as the layout of
    the HTML pages and the presentation of the entities is up to you. Please
    make sure that the HTML page displays at least the given query and the
    names, scores and descriptions of the given entities in the given order
    (descending sorted by scores).

     URL:
      http://<host>:<port>/search.html?q=angel
     RESPONSE:
      %QUERY%:
        angel
      %RESULT%:
       ["Angela Merkel;211;chancellor of Germany from 2005 to 2021",
        "Angelina Jolie;160;American actress (born 1975)",
        "angel;147;supernatural being or spirit in certain religions and\
                mythologies",
        "Angel Falls;91;waterfall in Venezuela; highest uninterrupted \
                waterfall in the world",
        "Angela Davis;73;American political activist, scholar, and author"
       ]

     URL:
      http://<host>:<port>/search.html?q=eyjaffjala
     RESPONSE:
      %QUERY%:
        eyjaffjala
      %RESULT%:
       ["Eyjafjallajökull;77;ice cap in Iceland covering the caldera of a \
                volcano",
        "Eyjafjallajökull;24;volcano in Iceland",
        "Eyjafjallajökull;8;2013 film by Alexandre Coffre"
       ]

     URL:
      http://<host>:<port>/search.html?q=The+hitschheiker+guide
     RESPONSE:
      %QUERY%:
       The hitschheiker guide
      %RESULT%:
       ["The Hitchhiker's Guide to the Galaxy pentalogy;45;1979-1992 series\
                of five books by Douglas Adams",
        "The Hitchhiker's Guide to the Galaxy;43;1979 book by Douglas Adams",
        "The Hitchhiker's Guide to the Galaxy;37;2005 film directed by Garth \
                Jennings",
        "The Hitchhiker's Guide to the Galaxy;8;1984 interactive fiction video\
                game",
        "The Hitchhiker's Guide to the Galaxy;8;BBC television series"
       ]
    """

    def __init__(
        self,
        port: int,
        qi: QGramIndex,
        db: str | None
    ) -> None:
        """

        Initializes a simple HTTP server with
        the given q-gram index and port.

        Using the database is optional (see task 4).

        """
        self.port = port
        self.qi = qi
        self.db = db
        self.status_response = {
            200: "OK",
            404: "Not found",
            403: "Forbidden",
            405: "Not allowed!"
        }

    def run(self) -> None:
        """

        Runs the server loop:
        Creates a socket, and then, in an infinite loop,
        waits for requests and processes them.

        """
        # creating the server socket and binding it to the port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        sock.bind(("0.0.0.0", self.port))
        sock.listen(1)

        # running the server loop
        # that listens and processes the clients requests
        while True:
            print(f"\nWaiting for incoming request on port: {self.port}")
            client_socket, client_address = sock.accept()
            print(f"Received request from: {client_address}")

            # Read bytes from client in rounds
            # initialising the byte array
            # because data from client is received as bytes
            request = b""

            while request.find(b"\r\n\r\n") == -1:  # b"\r\n\r\n" is new line
                # exit receiving data when empty line is received
                request += client_socket.recv(1024)

            # decode data back into string
            request = request.decode("utf-8")
            print(f"\nRequest from client\n{request}")

            # handle the request
            response = self.handle_request(request)

            # send the response
            print(f"Sending response to the client: {response}")
            client_socket.sendall(response)

            # close the connection
            client_socket.close()

    def handle_request(self, request):
        """
        Handle request and return the response in bytes
        """
        # parse request data from request
        # before HTTP and after GET
        """
        GET /search.html HTTP/1.1
        Host: localhost:8080
        User-Agent: curl/8.4.0
        Accept: */*
        """
        method, path, *_ = request.split(" ")  # split by new line
        # Only handle Get request
        if method != "GET":
            response = f"Server does not handle {method} request"
            response = response.encode("utf-8")
            headers = self.computer_headers(
                    405,
                    response,
                    None
            )
            return headers + response

        # check if path has query in it
        query_params = {}
        query_results = []
        if "?" in path:
            path, query = path.split('?')

            #  query params
            if len(query) > 1:
                for pair in query.split("&"):
                    key, value = pair.split("=")
                    query_params[key] = value

            # get the fuzzy matches for the query
            query = query_params["query"].replace('+', ' ')
            query_results = self.get_fuzzy_matches(query)

        # ensure path is relative to CWD/resources/
        filep = Path(__file__).parent.absolute().joinpath(
            Path("./resources/" + path)
        )

        # return 403 for file requested outside directory
        if not filep.is_file():
            response = f"You don't have access to the directory"
            response = response.encode("utf-8")
            headers = self.computer_headers(
                    403,
                    response,
                    None
            )
            return headers + response

        # guess MIME type
        mimetype, _ = mimetypes.guess_type(
            str(filep.resolve())
        ) or "text/plain"

        # reading the file requested by server
        # adding its content in response
        try:
            with open(filep, "r", encoding="utf-8") as file:
                response = file.read()

                # query and result found
                if len(query_results) > 0 and len(query_params["query"]) > 0:
                    print("i came here")
                    print(query_params)
                    print('sdasdsda', response)
                    response = response.replace("%QUERY%", query_params["query"])
                    response = response.replace("%RESULT%", str(query_results))
                else:
                    response = response.replace("%RESULT%", "")
                    response = response.replace("%QUERY%", "")

                headers = self.computer_headers(
                200,
                response,
                mimetype)
                # encode the response back to "utf-8"
                response = response.encode("utf-8")
                return headers + response
        except FileNotFoundError:
            headers = self.computer_headers(
                    404,
                    response,
                    None
                )
            response = f"{file} not found on the server"
            response = response.encode("utf-8")
            return headers + response



    def computer_headers(self, status_code, response, media_type):
        # computer the headers
        headers = f"HTTP/1.1 {status_code} {self.status_response[status_code]}\r\n"
        headers += f"Content-Length: {len(response)}\r\n"

        if media_type is not None:
            headers += f"Content-Type: {media_type}\r\n"
        headers += f"\r\n"
        headers = headers.encode("utf-8")

        return headers

    def get_fuzzy_matches(self, query):
        if query == "":
            return []
        results = []
        q_grams = 3
        delta = int(len(query) / (q_grams + 1))
        postings = self.qi.find_matches(query, delta)

        for syn_id, pedist in postings[:5]:
            infos = self.qi.get_infos(syn_id)
            syn, name, score, info = infos
            print(
                f"\n\033[1m{name}\033[0m (score={score}, ped={pedist}, "
                f"qid={info[0]}, via '{syn}'):\n{info[1]}"
            )
            results.append([
                f"\n\033[1m{name}\033[0m (score={score}, ped={pedist}, "
                f"qid={info[0]}, via '{syn}'):\n{info[1]}"
            ])
        return results

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "entities",
        type=str,
        help="path to entities file for q-gram index"
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
        "-db",
        "--database",
        type=str,
        default=None,
        help="path to sqlite3 database for SPARQL engine"
    )
    parser.add_argument(
        "-s",
        "--use-synonyms",
        action="store_true",
        help="whether to use synonyms for the q-gram index"
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """

    Builds a q-gram index from the given file
    and starts a server on the given port.

    """
    # Create a new q-gram index from the given file.
    print(f"Building q-gram index from file {args.entities}.")
    start = time.perf_counter()
    q = QGramIndex(args.q_grams, args.use_synonyms)
    q.build_from_file(args.entities)
    print(f"Done, took {(time.perf_counter() - start) * 1000:.1f}ms.")

    server = Server(
        args.port,
        q,
        args.database
    )
    print(
        f"Starting server on port {args.port}, go to "
        f"http://localhost:{args.port}/search.html"
    )
    server.run()


if __name__ == "__main__":
    main(parse_args())
