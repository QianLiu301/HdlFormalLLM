// shift_register_8bit — golden reference design
module shift_register_8bit (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       load,
    input  wire [7:0] d,
    input  wire       shift_en,
    input  wire       dir,
    input  wire       serial_in,
    output reg  [7:0] q,
    output wire       serial_out
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            q <= 8'h00;
        else if (load)
            q <= d;
        else if (shift_en) begin
            if (dir == 1'b0)
                q <= {serial_in, q[7:1]};   // BUG: dir semantics swapped
            else
                q <= {q[6:0], serial_in};   // BUG: dir semantics swapped
        end
    end
    assign serial_out = (dir == 1'b0) ? q[7] : q[0];
endmodule
