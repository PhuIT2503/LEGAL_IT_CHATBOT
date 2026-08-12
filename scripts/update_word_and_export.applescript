on run argv
    if (count of argv) is not 3 then error "Usage: input.docx output.docx output.pdf"
    set inputPath to item 1 of argv
    set outputDocxPath to item 2 of argv
    set outputPdfPath to item 3 of argv

    tell application "Microsoft Word"
        activate
        set thesisDocument to open file name inputPath

        -- Update ordinary fields first, then each generated list explicitly.
        set fieldCount to count of every field of thesisDocument
        repeat with fieldIndex from fieldCount to 1 by -1
            update field (field fieldIndex of thesisDocument)
        end repeat
        set tocCount to count of every table of contents of thesisDocument
        repeat with tocIndex from 1 to tocCount
            update (table of contents tocIndex of thesisDocument)
        end repeat
        set tofCount to count of every table of figures of thesisDocument
        repeat with tofIndex from 1 to tofCount
            update (table of figures tofIndex of thesisDocument)
        end repeat

        save as thesisDocument file name outputDocxPath file format format document default
        save as active document file name outputPdfPath file format format PDF
        close active document saving no
    end tell
end run
