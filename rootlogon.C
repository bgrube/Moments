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
	TColor* newColor = new TColor((Float_t)1, 1, 1, 1);
	newColor->SetName("white_background");
	const int fillColor = newColor->GetNumber();  // cannot use kWhite here
	gStyle->SetFillColor      (fillColor);
	gStyle->SetFrameFillColor (fillColor);
	gStyle->SetTitleColor     (fillColor);
	gStyle->SetTitleFillColor (fillColor);
	gStyle->SetStatColor      (fillColor);
	gStyle->SetLegendFillColor(fillColor);
	newColor = new TColor((Float_t)1, 1, 1, 0);  // cannot use TColor::GetColorTransparent as it checks for existing colors
	newColor->SetName("transparent_background");
	const int fullyTransparentColor = newColor->GetNumber();
	gStyle->SetPadColor   (fullyTransparentColor);
	gStyle->SetCanvasColor(fullyTransparentColor);

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
