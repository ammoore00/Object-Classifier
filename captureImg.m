function [img, captured] = captureImg(fig, url)

finish = false;
captured = false;

while ~finish
    %get images
    curImg = getCurrentWebread(url);
    
    k = get(fig,'CurrentCharacter');
    if k ~= '@'
        key = double(get(fig,'CurrentCharacter'));
        if key == 27 %esc key -- exits program
            finish=true;
        else
            if key == 13 %enter key -- captures image
                captured = true;
                img = curImg;
            end
        end
    end
end

close(fig);
end

