function dispImArray(Ima, width);

% Setwidth automatically if not passed in
if ~exist('width', 'var') || isempty(width) 
   width = round(sqrt(size(Ima, 2)));
end

% Compute rows, cols
[m n] = size(Ima);
height = n/width;

% Compute number of items to display
display_rows = floor(sqrt(m));
display_cols = ceil(m / display_rows);

% padding
pad = 1;

% Setup display
display_array = - ones(pad + display_rows * (height + pad), ...
                       pad + display_cols * (width + pad));

curr_ex = 1;
for j = 1:display_rows
	for i = 1:display_cols
		if curr_ex > m, 
		   break; 
		end
		max_val = max(abs(Ima(curr_ex, :)));
		display_array(pad + (j - 1) * (height + pad) + (1:height), ...
		              pad + (i - 1) * (width + pad) + (1:width)) = ...
						reshape(Ima(curr_ex, :), height, width) / max_val;
		curr_ex = curr_ex + 1;
	end
	if curr_ex > m, 
		break; 
	end
end

% Display Image
imagesc(display_array, [-1 1]);colormap gray;

% Do not show axis
axis image off; drawnow;

