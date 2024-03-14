document.querySelector("#searchField").addEventListener(
  "input", async function(){
    let query = document.querySelector("#searchField").value
    url = `/api/search?q=${query}`
    response = await fetch(url)
                .then(response => response.json())
    document.querySelector("#searchResults").innerHTML = `
    <div>${response.entities}</div>
    `
  }
)
