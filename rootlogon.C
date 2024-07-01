{
	gROOT->SetStyle("Modern");

	// no borders
	gStyle->SetCanvasBorderMode(0);
	gStyle->SetCanvasBorderSize(0);
	gStyle->SetPadBorderMode   (0);
	gStyle->SetFrameBorderMode (0);
	gStyle->SetTitleBorderSize (0);
	gStyle->SetLegendBorderSize(0);
	// all fill colors set to white + transparent pads and canvases
	// note that png export does write alpha channel
	// convert, for example, using
	// mutool draw -A 8 -r <dpi, e.g. 600> -c rgba -o <out.png> <in.pdf>
	TColor* newColor = new TColor(TColor::GetFreeColorIndex(), 1, 1, 1, "white_background", 1);
	int fillColor = newColor->GetNumber();  // if type is const int, ROOT 6.28+ crashes with 'cling::InvalidDerefException'
	gStyle->SetFillColor      (fillColor);
	gStyle->SetFrameFillColor (fillColor);
	gStyle->SetTitleColor     (fillColor);
	gStyle->SetTitleFillColor (fillColor);
	gStyle->SetStatColor      (fillColor);
	gStyle->SetLegendFillColor(fillColor);
	newColor = new TColor(TColor::GetFreeColorIndex(), 1, 1, 1, "transparent_background", 0);
	int fullyTransparentColor = newColor->GetNumber();
	gStyle->SetPadColor       (fullyTransparentColor);
	gStyle->SetCanvasColor    (fullyTransparentColor);
	gStyle->SetLegendFillColor(fullyTransparentColor);

	gStyle->SetOptTitle(true);
	gStyle->SetOptStat (true);
	gStyle->SetOptFit  (true);
	gStyle->SetOptDate (false);

	gStyle->SetPalette(kViridis);   // Viridis palette for 2D plots
	gStyle->SetNumberContours(100);

	gStyle->SetHistMinimumZero(true);
	TGaxis::SetMaxDigits(3);  // restrict the number of digits in axis tick labels

	// gStyle->SetPaperSize(kA4);   // set A4 paper size
	gStyle->SetPaperSize(TStyle::kUSLetter);  // set letter paper size
}
