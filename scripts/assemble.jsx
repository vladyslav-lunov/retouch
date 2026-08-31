// Складання шарів у Photoshop.
// Файл -> Сценарії -> Обзор -> цей файл, далі вибрати теку out/.
//
// Тимчасове рішення, поки конвеєр не пише PSD напряму (spec.md, v0.4).

#target photoshop

function main() {
    var dir = Folder.selectDialog("Тека з результатом (out/)");
    if (!dir) return;

    var files = dir.getFiles(function (f) {
        return f instanceof File && /_\d\d_.*\.(tif|png)$/i.test(f.name);
    });
    if (!files.length) { alert("Шарів не знайдено."); return; }
    files.sort(function (a, b) { return a.name < b.name ? -1 : 1; });

    var doc = app.open(files[0]);           // _00_base
    doc.layers[0].name = "base";

    for (var i = 1; i < files.length; i++) {
        if (/_99_flat/i.test(files[i].name)) continue;   // зведений не потрібен
        var src = app.open(files[i]);
        src.selection.selectAll();
        src.selection.copy();
        src.close(SaveOptions.DONOTSAVECHANGES);

        app.activeDocument = doc;
        doc.paste();
        doc.activeLayer.name = files[i].name
            .replace(/^.*_\d\d_/, "").replace(/\.(png|tif)$/i, "");
    }
    alert("Готово. Шари зверху вниз — у порядку застосування.");
}

main();
