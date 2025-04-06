const id = "1LHbOkccCjHTQudFQk7tMh-vhk24EU_Im"

let clouds = {}

$(() => {
    // get names and aliases (hash) from excel file on google drive)
    $.ajax({
        type: "GET",
        url: "https://docs.google.com/spreadsheets/d/" + id + "/export?format=tsv&id=" + id + "&gid=1963659479",
        dataType: "text",
        success: text => {
            const lines = text.split('\r\n');
            for (let line of lines) {
                let info = line.split('\t')
                clouds[info[0]] = info[1]
            }
        }
    })

    // set callback on search bar when clicked or input is changed
    $('#search').on('input click', _ => filter(10))

    // set callback on image function to display every time a new image gets loaded
    let cloud = $('#cloud')
    cloud.on('load', _ => cloud.show())
})

function filter(limit) {
    // retrive cards div and search text
    let text = $('#search').val().trim()
    let cards = $('#cards')
    let items = []
    // if the text is not empty, match it with the names
    if (text.length > 0) {
        items = Object.keys(clouds)
            .filter(name => name.toLowerCase().indexOf(text) !== -1)
            .map(name => `<div class="card-item" onclick="select('${name}')">${name}</div>`)
    }
    // if there is at least one item, show the cards div and hide the wordcloud, otherwise hide the cards
    if (items.length > 0) {
        cards.show()
        // if there are too many matches, show the first <limit> (if available) and add a last '...' item to expand
        if (limit != null && items.length > limit) {
            items = items.slice(0, limit)
            items[limit - 1] = '<div class="card-item" onclick="filter(null)">...</div>'
        }
        cards.html(items.join('\n'))
        $('#cloud').hide()
    } else {
        cards.hide()
    }
}

function select(name) {
    // hide the cards div, empty the search bar, and show the wordcloud by retrieving the alias from the name
    $('#cards').hide()
    $('#search').val('')
    $('#cloud').attr("src", `res/clouds/${clouds[name]}.png`)
}