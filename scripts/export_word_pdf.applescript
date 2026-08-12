on run argv
    if (count of argv) is not 2 then error "Usage: input.docx output.pdf"
    set inputPath to item 1 of argv
    set outputPdfPath to item 2 of argv
    tell application "Microsoft Word"
        activate
        repeat while (count of documents) > 0
            close document 1 saving no
        end repeat
        open file name inputPath
        save as active document file name outputPdfPath file format format PDF
        close active document saving no
    end tell
end run
